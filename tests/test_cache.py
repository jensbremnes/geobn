"""Tests for disk caching in WCSSource, URLSource and PointGridSource."""
from __future__ import annotations

import json
import os
import time
from datetime import timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from affine import Affine

from geobn._types import RasterData
from geobn.grid import GridSpec
from geobn.sources._cache import _load_cached, _make_cache_path, _save_cached
from geobn.sources.point_grid_source import PointGridSource
from geobn.sources.url_source import URLSource
from geobn.sources.wcs_source import WCSSource


def _age_cache_entry(cache_dir, seconds):
    """Backdate every cache entry in *cache_dir* by *seconds*."""
    for meta_path in cache_dir.glob("*.json"):
        meta = json.loads(meta_path.read_text())
        meta["fetched_at"] = time.time() - seconds
        meta_path.write_text(json.dumps(meta))


@pytest.fixture
def small_grid():
    return GridSpec(crs="EPSG:4326", transform=Affine(0.1, 0, 5.0, 0, -0.1, 62.0), shape=(5, 5))


def _make_tiff_bytes(arr):
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    H, W = arr.shape
    with MemoryFile() as mf:
        with mf.open(
            driver="GTiff", height=H, width=W, count=1,
            dtype=np.float32, crs="EPSG:4326",
            transform=from_bounds(5, 60, 6, 62, W, H),
        ) as ds:
            ds.write(arr.astype(np.float32), 1)
        return mf.read()


class TestWCSSourceCache:
    def test_cache_miss_fetches_and_saves(self, small_grid, tmp_path):
        tiff = _make_tiff_bytes(np.ones((5, 5), np.float32) * 42.0)
        mock_resp = MagicMock(ok=True, content=tiff)
        with patch("requests.get", return_value=mock_resp) as mock_get:
            src = WCSSource("http://x.com/wcs", "layer", cache_dir=tmp_path)
            data = src.fetch(grid=small_grid)
        mock_get.assert_called_once()
        assert data.array.mean() == pytest.approx(42.0)
        # Cache files written to disk
        assert any(tmp_path.glob("*.npy"))
        assert any(tmp_path.glob("*.json"))

    def test_cache_hit_skips_network(self, small_grid, tmp_path):
        tiff = _make_tiff_bytes(np.ones((5, 5), np.float32) * 7.0)
        mock_resp = MagicMock(ok=True, content=tiff)
        with patch("requests.get", return_value=mock_resp) as mock_get:
            src = WCSSource("http://x.com/wcs", "layer", cache_dir=tmp_path)
            src.fetch(grid=small_grid)       # populate cache
            data = src.fetch(grid=small_grid)  # should hit cache
        assert mock_get.call_count == 1      # only one HTTP call total
        assert data.array.mean() == pytest.approx(7.0)

    def test_corrupt_cache_refetches(self, small_grid, tmp_path):
        tiff = _make_tiff_bytes(np.ones((5, 5), np.float32))
        mock_resp = MagicMock(ok=True, content=tiff)
        with patch("requests.get", return_value=mock_resp):
            src = WCSSource("http://x.com/wcs", "layer", cache_dir=tmp_path)
            src.fetch(grid=small_grid)  # populate
        # Corrupt the .npy file
        for f in tmp_path.glob("*.npy"):
            f.write_bytes(b"not a numpy file")
        with patch("requests.get", return_value=mock_resp) as mock_get2:
            src.fetch(grid=small_grid)
        mock_get2.assert_called_once()  # re-fetched after corrupt

    def test_no_cache_dir_does_not_write(self, small_grid, tmp_path):
        tiff = _make_tiff_bytes(np.ones((5, 5), np.float32))
        mock_resp = MagicMock(ok=True, content=tiff)
        with patch("requests.get", return_value=mock_resp):
            src = WCSSource("http://x.com/wcs", "layer")  # no cache_dir
            src.fetch(grid=small_grid)
        assert not any(tmp_path.glob("*.npy"))

    def test_different_grids_have_different_cache_entries(self, tmp_path):
        grid_a = GridSpec(crs="EPSG:4326", transform=Affine(0.1, 0, 5.0, 0, -0.1, 62.0), shape=(5, 5))
        grid_b = GridSpec(crs="EPSG:4326", transform=Affine(0.1, 0, 10.0, 0, -0.1, 65.0), shape=(5, 5))
        tiff_a = _make_tiff_bytes(np.full((5, 5), 1.0, np.float32))
        tiff_b = _make_tiff_bytes(np.full((5, 5), 2.0, np.float32))
        responses = iter([
            MagicMock(ok=True, content=tiff_a),
            MagicMock(ok=True, content=tiff_b),
        ])
        with patch("requests.get", side_effect=lambda *a, **kw: next(responses)) as mock_get:
            src = WCSSource("http://x.com/wcs", "layer", cache_dir=tmp_path)
            data_a = src.fetch(grid=grid_a)
            data_b = src.fetch(grid=grid_b)
        assert mock_get.call_count == 2
        assert data_a.array.mean() == pytest.approx(1.0)
        assert data_b.array.mean() == pytest.approx(2.0)
        assert len(list(tmp_path.glob("*.npy"))) == 2


    def test_default_cache_key_unchanged(self, small_grid):
        """Default options keep the pre-0.8 key, so existing cache entries stay valid."""
        src = WCSSource("http://x.com/wcs", "layer")
        lon_min, lat_min, lon_max, lat_max = small_grid.extent_wgs84()
        assert src._cache_key(small_grid) == {
            "url": "http://x.com/wcs", "layer": "layer", "version": "2.0.1",
            "lon_min": round(lon_min, 8), "lat_min": round(lat_min, 8),
            "lon_max": round(lon_max, 8), "lat_max": round(lat_max, 8),
            "H": 5, "W": 5,
        }

    def test_cache_ttl_is_not_part_of_the_key(self, small_grid):
        """Freshness is a property of the entry, not of what was requested."""
        plain = WCSSource("http://x.com/wcs", "layer")
        with_ttl = WCSSource("http://x.com/wcs", "layer", cache_ttl=timedelta(hours=6))
        assert plain._cache_key(small_grid) == with_ttl._cache_key(small_grid)

    @pytest.mark.parametrize("kwargs", [
        {"extra_subsets": ['time("2023-01-01T00:00:00.000Z")']},
        {"format": "application/x-geotiff"},
        {"axis_labels": ("lon", "lat")},
    ])
    def test_request_options_get_own_cache_entry(self, small_grid, tmp_path, kwargs):
        tiff_a = _make_tiff_bytes(np.full((5, 5), 1.0, np.float32))
        tiff_b = _make_tiff_bytes(np.full((5, 5), 2.0, np.float32))
        responses = iter([
            MagicMock(ok=True, content=tiff_a),
            MagicMock(ok=True, content=tiff_b),
        ])
        with patch("requests.get", side_effect=lambda *a, **kw: next(responses)) as mock_get:
            data_a = WCSSource("http://x.com/wcs", "layer", cache_dir=tmp_path).fetch(grid=small_grid)
            data_b = WCSSource("http://x.com/wcs", "layer", cache_dir=tmp_path, **kwargs).fetch(grid=small_grid)
        assert mock_get.call_count == 2
        assert data_a.array.mean() == pytest.approx(1.0)
        assert data_b.array.mean() == pytest.approx(2.0)
        assert len(list(tmp_path.glob("*.npy"))) == 2

    def test_valid_range_shares_one_cache_entry(self, small_grid, tmp_path):
        """The cached array is the one the server sent, so the range is applied on top."""
        tiff = _make_tiff_bytes(np.full((5, 5), -9999.0, np.float32))
        with patch("requests.get", return_value=MagicMock(ok=True, content=tiff)) as mock_get:
            unmasked = WCSSource(
                "http://x.com/wcs", "layer", cache_dir=tmp_path,
            ).fetch(grid=small_grid)
            masked = WCSSource(
                "http://x.com/wcs", "layer", cache_dir=tmp_path, valid_range=(0.0, None),
            ).fetch(grid=small_grid)

        assert mock_get.call_count == 1
        assert len(list(tmp_path.glob("*.npy"))) == 1
        assert unmasked.array.mean() == pytest.approx(-9999.0)
        assert np.all(np.isnan(masked.array))


class TestURLSourceCache:
    def test_cache_hit_skips_network(self, tmp_path):
        tiff = _make_tiff_bytes(np.ones((5, 5), np.float32) * 3.0)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.content = tiff
        with patch("requests.get", return_value=mock_resp) as mock_get:
            src = URLSource("http://example.com/dem.tif", cache_dir=tmp_path)
            src.fetch()           # miss → saves
            data = src.fetch()    # hit → no HTTP call
        assert mock_get.call_count == 1
        assert data.array.mean() == pytest.approx(3.0)

    def test_cache_miss_fetches_and_saves(self, tmp_path):
        tiff = _make_tiff_bytes(np.ones((5, 5), np.float32) * 5.0)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.content = tiff
        with patch("requests.get", return_value=mock_resp) as mock_get:
            src = URLSource("http://example.com/dem.tif", cache_dir=tmp_path)
            data = src.fetch()
        mock_get.assert_called_once()
        assert data.array.mean() == pytest.approx(5.0)
        assert any(tmp_path.glob("*.npy"))

    def test_no_cache_dir_does_not_write(self, tmp_path):
        tiff = _make_tiff_bytes(np.ones((5, 5), np.float32))
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.content = tiff
        with patch("requests.get", return_value=mock_resp):
            src = URLSource("http://example.com/dem.tif")  # no cache_dir
            src.fetch()
        assert not any(tmp_path.glob("*.npy"))

    def test_corrupt_cache_refetches(self, tmp_path):
        tiff = _make_tiff_bytes(np.ones((5, 5), np.float32) * 9.0)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.content = tiff
        with patch("requests.get", return_value=mock_resp):
            src = URLSource("http://example.com/dem.tif", cache_dir=tmp_path)
            src.fetch()  # populate
        for f in tmp_path.glob("*.npy"):
            f.write_bytes(b"garbage")
        with patch("requests.get", return_value=mock_resp) as mock_get2:
            data = src.fetch()
        mock_get2.assert_called_once()
        assert data.array.mean() == pytest.approx(9.0)


class TestCacheTransformNone:
    """_save_cached / _load_cached must handle transform=None without crashing."""

    def test_save_and_load_with_transform_none(self, tmp_path):
        data = RasterData(
            array=np.array([[1.0, 2.0]], dtype=np.float32),
            crs=None,
            transform=None,
        )
        cache_path = _make_cache_path(tmp_path, {"key": "no-transform"})
        _save_cached(cache_path, data)
        loaded = _load_cached(cache_path)
        assert loaded is not None
        np.testing.assert_array_equal(loaded.array, data.array)
        assert loaded.crs is None
        assert loaded.transform is None

    def test_save_and_load_with_real_transform(self, tmp_path):
        t = Affine(0.1, 0, 5.0, 0, -0.1, 62.0)
        data = RasterData(
            array=np.ones((3, 3), dtype=np.float32),
            crs="EPSG:4326",
            transform=t,
        )
        cache_path = _make_cache_path(tmp_path, {"key": "with-transform"})
        _save_cached(cache_path, data)
        loaded = _load_cached(cache_path)
        assert loaded is not None
        assert loaded.transform == t
        assert loaded.crs == "EPSG:4326"


class TestCacheVersion:
    def test_cache_version_is_part_of_the_key(self, tmp_path):
        """Bumping _CACHE_VERSION must change the path so stale entries are not reused."""
        key = {"url": "http://example.com/dem.tif"}
        current = _make_cache_path(tmp_path, key)
        with patch("geobn.sources._cache._CACHE_VERSION", 1):
            old = _make_cache_path(tmp_path, key)
        assert current != old


# ---------------------------------------------------------------------------
# Cache freshness
# ---------------------------------------------------------------------------

class TestCacheTTL:
    def _entry(self, tmp_path, value=1.0):
        data = RasterData(
            array=np.full((3, 3), value, dtype=np.float32),
            crs="EPSG:4326",
            transform=Affine(0.1, 0, 5.0, 0, -0.1, 62.0),
        )
        cache_path = _make_cache_path(tmp_path, {"key": "ttl"})
        _save_cached(cache_path, data)
        return cache_path

    def test_save_records_fetched_at(self, tmp_path):
        cache_path = self._entry(tmp_path)
        meta = json.loads(cache_path.with_suffix(".json").read_text())
        assert meta["fetched_at"] == pytest.approx(time.time(), abs=60)

    def test_fresh_entry_is_a_hit(self, tmp_path):
        cache_path = self._entry(tmp_path)
        assert _load_cached(cache_path, ttl=3600) is not None

    def test_expired_entry_is_a_miss(self, tmp_path):
        cache_path = self._entry(tmp_path)
        _age_cache_entry(tmp_path, seconds=7200)
        assert _load_cached(cache_path, ttl=3600) is None

    def test_entry_within_ttl_is_a_hit(self, tmp_path):
        cache_path = self._entry(tmp_path)
        _age_cache_entry(tmp_path, seconds=600)
        assert _load_cached(cache_path, ttl=3600) is not None

    def test_no_ttl_never_expires(self, tmp_path):
        cache_path = self._entry(tmp_path)
        _age_cache_entry(tmp_path, seconds=86_400 * 365)
        assert _load_cached(cache_path) is not None

    def test_missing_fetched_at_falls_back_to_mtime(self, tmp_path):
        """Entries written without a timestamp are dated by the array file."""
        cache_path = self._entry(tmp_path)
        meta_path = cache_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        del meta["fetched_at"]
        meta_path.write_text(json.dumps(meta))

        assert _load_cached(cache_path, ttl=3600) is not None      # just written
        old = time.time() - 7200
        os.utime(cache_path, (old, old))
        assert _load_cached(cache_path, ttl=3600) is None


class TestCacheTTLOnSources:
    def test_expired_entry_refetches(self, small_grid, tmp_path):
        tiff_a = _make_tiff_bytes(np.full((5, 5), 1.0, np.float32))
        tiff_b = _make_tiff_bytes(np.full((5, 5), 2.0, np.float32))
        responses = iter([MagicMock(ok=True, content=tiff_a), MagicMock(ok=True, content=tiff_b)])
        src = WCSSource("http://x.com/wcs", "layer", cache_dir=tmp_path, cache_ttl=3600)

        with patch("requests.get", side_effect=lambda *a, **kw: next(responses)) as mock_get:
            first = src.fetch(grid=small_grid)
            _age_cache_entry(tmp_path, seconds=7200)
            second = src.fetch(grid=small_grid)

        assert mock_get.call_count == 2
        assert first.array.mean() == pytest.approx(1.0)
        assert second.array.mean() == pytest.approx(2.0)

    def test_fresh_entry_skips_network(self, small_grid, tmp_path):
        tiff = _make_tiff_bytes(np.full((5, 5), 1.0, np.float32))
        src = WCSSource("http://x.com/wcs", "layer", cache_dir=tmp_path, cache_ttl=3600)

        with patch("requests.get", return_value=MagicMock(ok=True, content=tiff)) as mock_get:
            src.fetch(grid=small_grid)
            src.fetch(grid=small_grid)

        assert mock_get.call_count == 1

    def test_stale_entry_served_with_warning_when_refetch_fails(self, small_grid, tmp_path):
        tiff = _make_tiff_bytes(np.full((5, 5), 7.0, np.float32))
        src = WCSSource("http://x.com/wcs", "layer", cache_dir=tmp_path, cache_ttl=3600)

        with patch("requests.get", return_value=MagicMock(ok=True, content=tiff)):
            src.fetch(grid=small_grid)
        _age_cache_entry(tmp_path, seconds=7200)

        with patch("requests.get", side_effect=OSError("network down")):
            with pytest.warns(UserWarning, match="using cached data"):
                data = src.fetch(grid=small_grid)

        assert data.array.mean() == pytest.approx(7.0)

    def test_failure_without_any_entry_raises(self, small_grid, tmp_path):
        src = WCSSource("http://x.com/wcs", "layer", cache_dir=tmp_path, cache_ttl=3600)
        with patch("requests.get", side_effect=OSError("network down")):
            with pytest.raises(OSError, match="network down"):
                src.fetch(grid=small_grid)

    def test_url_source_honours_ttl(self, tmp_path):
        tiff_a = _make_tiff_bytes(np.full((5, 5), 1.0, np.float32))
        tiff_b = _make_tiff_bytes(np.full((5, 5), 2.0, np.float32))
        responses = iter([
            MagicMock(raise_for_status=lambda: None, content=tiff_a),
            MagicMock(raise_for_status=lambda: None, content=tiff_b),
        ])
        src = URLSource("http://x.com/dem.tif", cache_dir=tmp_path, cache_ttl=timedelta(hours=1))

        with patch("requests.get", side_effect=lambda *a, **kw: next(responses)) as mock_get:
            first = src.fetch()
            _age_cache_entry(tmp_path, seconds=7200)
            second = src.fetch()

        assert mock_get.call_count == 2
        assert first.array.mean() == pytest.approx(1.0)
        assert second.array.mean() == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("ttl", "seconds"),
        [(timedelta(hours=6), 21_600), (3600, 3600.0), (1.5, 1.5), (None, None)],
    )
    def test_cache_ttl_accepted_forms(self, ttl, seconds):
        src = URLSource("http://x.com/dem.tif", cache_ttl=ttl)
        assert src._cache_ttl == seconds

    @pytest.mark.parametrize("bad", [-1, -timedelta(hours=1), float("nan"), "6h", object()])
    def test_invalid_cache_ttl_raises(self, bad):
        with pytest.raises(ValueError, match="cache_ttl"):
            URLSource("http://x.com/dem.tif", cache_ttl=bad)


class TestPointGridSourceCache:
    def _counting_fn(self, calls):
        def fn(lat, lon):
            calls.append((lat, lon))
            return 1.0
        return fn

    def test_cache_hit_skips_all_calls(self, small_grid, tmp_path):
        calls: list = []
        src = PointGridSource(
            fn=self._counting_fn(calls), sample_points=3, delay=0.0,
            name="wave_height", cache_dir=tmp_path,
        )
        first = src.fetch(grid=small_grid)
        assert len(calls) == 9
        second = src.fetch(grid=small_grid)
        assert len(calls) == 9                      # no further sampling
        np.testing.assert_array_equal(first.array, second.array)

    def test_expired_entry_resamples(self, small_grid, tmp_path):
        calls: list = []
        src = PointGridSource(
            fn=self._counting_fn(calls), sample_points=2, delay=0.0,
            name="wave_height", cache_dir=tmp_path, cache_ttl=3600,
        )
        src.fetch(grid=small_grid)
        _age_cache_entry(tmp_path, seconds=7200)
        src.fetch(grid=small_grid)
        assert len(calls) == 8                      # sampled twice

    def test_different_names_get_their_own_entry(self, small_grid, tmp_path):
        for name in ("wave_height", "wind_speed"):
            PointGridSource(
                fn=lambda lat, lon: 1.0, sample_points=2, delay=0.0,
                name=name, cache_dir=tmp_path,
            ).fetch(grid=small_grid)
        assert len(list(tmp_path.glob("*.npy"))) == 2

    def test_different_sample_points_get_their_own_entry(self, small_grid, tmp_path):
        for n in (2, 3):
            PointGridSource(
                fn=lambda lat, lon: 1.0, sample_points=n, delay=0.0,
                name="wave_height", cache_dir=tmp_path,
            ).fetch(grid=small_grid)
        assert len(list(tmp_path.glob("*.npy"))) == 2

    def test_different_grids_get_their_own_entry(self, small_grid, tmp_path):
        other = GridSpec(
            crs="EPSG:4326", transform=Affine(0.1, 0, 20.0, 0, -0.1, 62.0), shape=(5, 5)
        )
        src = PointGridSource(
            fn=lambda lat, lon: 1.0, sample_points=2, delay=0.0,
            name="wave_height", cache_dir=tmp_path,
        )
        src.fetch(grid=small_grid)
        src.fetch(grid=other)
        assert len(list(tmp_path.glob("*.npy"))) == 2

    def test_single_point_broadcast_round_trips(self, small_grid, tmp_path):
        """sample_points=1 returns crs=None/transform=None, which must cache too."""
        src = PointGridSource(
            fn=lambda lat, lon: 7.5, sample_points=1, delay=0.0,
            name="wind_speed", cache_dir=tmp_path,
        )
        src.fetch(grid=small_grid)
        cached = src.fetch(grid=small_grid)
        assert cached.crs is None
        assert cached.transform is None
        assert float(cached.array[0, 0]) == pytest.approx(7.5)

    def test_cache_dir_without_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="name"):
            PointGridSource(fn=lambda lat, lon: 1.0, cache_dir=tmp_path)

    def test_no_cache_dir_does_not_write(self, small_grid, tmp_path):
        src = PointGridSource(fn=lambda lat, lon: 1.0, sample_points=2, delay=0.0)
        src.fetch(grid=small_grid)
        assert list(tmp_path.glob("*")) == []
