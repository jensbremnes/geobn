"""Tests for disk caching in WCSSource and URLSource."""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from affine import Affine

from geobn.grid import GridSpec
from geobn.sources.url_source import URLSource
from geobn.sources.wcs_source import WCSSource


@pytest.fixture
def small_grid():
    return GridSpec(crs="EPSG:4326", transform=Affine(0.1, 0, 5.0, 0, -0.1, 62.0), shape=(5, 5))


def _make_tiff_bytes(arr):
    rasterio = pytest.importorskip("rasterio")
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
        pytest.importorskip("rasterio")
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
        pytest.importorskip("rasterio")
        tiff = _make_tiff_bytes(np.ones((5, 5), np.float32) * 7.0)
        mock_resp = MagicMock(ok=True, content=tiff)
        with patch("requests.get", return_value=mock_resp) as mock_get:
            src = WCSSource("http://x.com/wcs", "layer", cache_dir=tmp_path)
            src.fetch(grid=small_grid)       # populate cache
            data = src.fetch(grid=small_grid)  # should hit cache
        assert mock_get.call_count == 1      # only one HTTP call total
        assert data.array.mean() == pytest.approx(7.0)

    def test_corrupt_cache_refetches(self, small_grid, tmp_path):
        pytest.importorskip("rasterio")
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
        pytest.importorskip("rasterio")
        tiff = _make_tiff_bytes(np.ones((5, 5), np.float32))
        mock_resp = MagicMock(ok=True, content=tiff)
        with patch("requests.get", return_value=mock_resp):
            src = WCSSource("http://x.com/wcs", "layer")  # no cache_dir
            src.fetch(grid=small_grid)
        assert not any(tmp_path.glob("*.npy"))

    def test_different_grids_have_different_cache_entries(self, tmp_path):
        pytest.importorskip("rasterio")
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


class TestURLSourceCache:
    def test_cache_hit_skips_network(self, tmp_path):
        pytest.importorskip("rasterio")
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
        pytest.importorskip("rasterio")
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
        pytest.importorskip("rasterio")
        tiff = _make_tiff_bytes(np.ones((5, 5), np.float32))
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.content = tiff
        with patch("requests.get", return_value=mock_resp):
            src = URLSource("http://example.com/dem.tif")  # no cache_dir
            src.fetch()
        assert not any(tmp_path.glob("*.npy"))

    def test_corrupt_cache_refetches(self, tmp_path):
        pytest.importorskip("rasterio")
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
