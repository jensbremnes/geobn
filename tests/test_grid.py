"""Tests for GridSpec and alignment logic."""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer

from geobn._types import RasterData
from geobn.grid import GridSpec, _bilinear_resample, _pixel_size_m, _reproject, align_to_grid


class TestGridSpec:
    def test_from_raster_data(self, slope_array, reference_transform):
        data = RasterData(array=slope_array, crs="EPSG:4326", transform=reference_transform)
        grid = GridSpec.from_raster_data(data)
        assert grid.shape == (10, 10)
        assert grid.crs == "EPSG:4326"
        assert grid.transform == reference_transform

    def test_from_raster_data_no_crs_raises(self, slope_array):
        data = RasterData(array=slope_array, crs=None, transform=None)
        with pytest.raises(ValueError, match="no CRS"):
            GridSpec.from_raster_data(data)

    def test_from_params(self):
        grid = GridSpec.from_params("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        assert grid.shape == (10, 10)
        assert grid.transform.a == pytest.approx(0.1)
        assert grid.transform.e == pytest.approx(-0.1)

    def test_from_params_inverted_x_raises(self):
        with pytest.raises(ValueError, match="xmin < xmax"):
            GridSpec.from_params("EPSG:4326", 0.1, (1.0, 49.0, 0.0, 50.0))

    def test_from_params_inverted_y_raises(self):
        with pytest.raises(ValueError, match="ymin < ymax"):
            GridSpec.from_params("EPSG:4326", 0.1, (0.0, 50.0, 1.0, 49.0))

    def test_from_params_equal_extent_raises(self):
        with pytest.raises(ValueError):
            GridSpec.from_params("EPSG:4326", 0.1, (0.0, 49.0, 0.0, 50.0))

    def test_extent_wgs84_identity(self, slope_array, reference_transform):
        """A grid in EPSG:4326 should return its own extent."""
        data = RasterData(array=slope_array, crs="EPSG:4326", transform=reference_transform)
        grid = GridSpec.from_raster_data(data)
        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        assert lon_min == pytest.approx(0.0, abs=0.01)
        assert lat_max == pytest.approx(50.0, abs=0.01)

    def test_extent_wgs84_covers_curved_edges(self):
        """Edge midpoints bulge past the corners in a large projected grid."""
        grid = GridSpec(
            crs="EPSG:3035",
            transform=Affine(1000, 0, 4_000_000, 0, -1000, 4_000_000),
            shape=(1000, 1000),
        )
        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        to_wgs84 = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
        H, W = grid.shape
        corners = [(0, 0), (W, 0), (0, H), (W, H)]
        midpoints = [(W / 2, 0), (W, H / 2), (W / 2, H), (0, H / 2)]
        corner_lons, corner_lats = to_wgs84.transform(*zip(*(grid.transform * p for p in corners)))
        mid_lons, mid_lats = to_wgs84.transform(*zip(*(grid.transform * p for p in midpoints)))
        # Some edge midpoint lies outside the corner-only box...
        assert max(mid_lats) > max(corner_lats) or min(mid_lats) < min(corner_lats)
        # ...but every midpoint lies inside the densified box
        for lon, lat in zip(mid_lons, mid_lats):
            assert lon_min <= lon <= lon_max
            assert lat_min <= lat <= lat_max

    @pytest.mark.parametrize("crs, pole", [("EPSG:3413", 90.0), ("EPSG:3031", -90.0)])
    def test_extent_wgs84_polar_grid_includes_pole(self, crs, pole):
        grid = GridSpec.from_params(crs, 10_000, (-500_000, -500_000, 500_000, 500_000))
        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        assert (lon_min, lon_max) == (-180.0, 180.0)
        if pole > 0:
            assert lat_max == 90.0
            assert lat_min == pytest.approx(83.5, abs=0.2)
        else:
            assert lat_min == -90.0
            assert lat_max == pytest.approx(-83.5, abs=0.2)

    def test_extent_wgs84_utm_without_pole(self):
        """The pole projects outside a UTM grid, so the box stays local."""
        grid = GridSpec(
            crs="EPSG:32633", transform=Affine(10, 0, 700_000, 0, -10, 7_750_000), shape=(2000, 2000)
        )
        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        assert 18 < lon_min < lon_max < 21
        assert 69 < lat_min < lat_max < 70


class TestPixelSizeM:
    def test_projected_metres(self):
        grid = GridSpec(
            crs="EPSG:32632", transform=Affine(10, 0, 500_000, 0, -10, 6_650_000), shape=(20, 20)
        )
        # UTM scale factor near the central meridian is ~0.9996
        assert _pixel_size_m(grid) == pytest.approx(10.0, rel=1e-2)

    def test_degrees_at_equator(self):
        grid = GridSpec(crs="EPSG:4326", transform=Affine(0.001, 0, 0.0, 0, -0.001, 0.01), shape=(20, 20))
        assert _pixel_size_m(grid) == pytest.approx(111.0, rel=1e-2)

    def test_degrees_at_60n_uses_geometric_mean(self):
        grid = GridSpec(crs="EPSG:4326", transform=Affine(0.001, 0, 9.0, 0, -0.001, 60.01), shape=(20, 20))
        assert _pixel_size_m(grid) == pytest.approx(np.sqrt(55.8 * 111.4), rel=1e-2)


class TestAlignToGrid:
    def test_constant_source_broadcast(self, slope_array, reference_transform):
        data = RasterData(array=np.array([[7.5]], dtype=np.float32), crs=None, transform=None)
        grid = GridSpec(crs="EPSG:4326", transform=reference_transform, shape=(10, 10))
        result = align_to_grid(data, grid)
        assert result.shape == (10, 10)
        assert np.all(result == pytest.approx(7.5))

    def test_no_crs_mismatched_shape_raises(self, reference_transform):
        """A no-CRS array that is neither 1×1 nor grid-shaped must not be broadcast."""
        data = RasterData(array=np.ones((4, 4), dtype=np.float32), crs=None, transform=None)
        grid = GridSpec(crs="EPSG:4326", transform=reference_transform, shape=(10, 10))
        with pytest.raises(ValueError, match="without CRS"):
            align_to_grid(data, grid)

    def test_identity_passthrough(self, slope_array, reference_transform):
        data = RasterData(array=slope_array, crs="EPSG:4326", transform=reference_transform)
        grid = GridSpec(crs="EPSG:4326", transform=reference_transform, shape=(10, 10))
        result = align_to_grid(data, grid)
        np.testing.assert_array_almost_equal(result, slope_array)

    def test_same_crs_resample(self, reference_transform):
        """Upsample a 5×5 array to 10×10 in the same CRS."""
        src = np.arange(25, dtype=np.float32).reshape(5, 5)
        src_transform = Affine(0.2, 0, 0.0, 0, -0.2, 50.0)  # 0.2° pixels
        data = RasterData(array=src, crs="EPSG:4326", transform=src_transform)

        dst_transform = Affine(0.1, 0, 0.0, 0, -0.1, 50.0)  # 0.1° pixels
        grid = GridSpec(crs="EPSG:4326", transform=dst_transform, shape=(10, 10))

        result = align_to_grid(data, grid)
        assert result.shape == (10, 10)
        # Centre of resampled grid should match centre of source grid
        assert not np.all(np.isnan(result))


class TestReproject:
    def test_same_crs_identity(self):
        """Reprojecting to the same grid should return pixel-identical data."""
        src = np.arange(9, dtype=np.float32).reshape(3, 3)
        t = Affine(1.0, 0, 0.0, 0, -1.0, 3.0)
        result = _reproject(src, "EPSG:4326", t, "EPSG:4326", t, (3, 3))
        np.testing.assert_array_almost_equal(result, src)

    def test_upsample_doubles_resolution(self):
        """Upsampling a 2×2 array to 4×4 should produce a valid, finite result."""
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        src_transform = Affine(1.0, 0, 0.0, 0, -1.0, 2.0)  # 1° pixels, 2×2
        dst_transform = Affine(0.5, 0, 0.0, 0, -0.5, 2.0)  # 0.5° pixels, 4×4
        result = _reproject(src, "EPSG:4326", src_transform, "EPSG:4326", dst_transform, (4, 4))
        assert result.shape == (4, 4)
        assert not np.any(np.isnan(result))
        # Bilinear interpolation at pixel [1,1] (world centre 0.75°, 1.25°):
        # maps to src fractional coords (0.75, 0.75) → interpolated value is 1.75
        assert result[1, 1] == pytest.approx(1.75, abs=1e-4)

    def test_cross_crs_reprojection_preserves_values(self):
        """Reprojecting a uniform array across CRS should preserve values in overlapping pixels."""
        # 100×100 km uniform grid in EPSG:32632 (UTM zone 32N, central meridian 9°E).
        # Centred at easting 500000 / northing 6651444 ≈ (9°E, 60°N).
        src = np.ones((100, 100), dtype=np.float32) * 42.0
        src_transform = Affine(1000.0, 0, 450000.0, 0, -1000.0, 6701444.0)
        # 5×5 destination grid in EPSG:4326 covering ~9°E, 60°N
        dst_transform = Affine(0.01, 0, 8.97, 0, -0.01, 60.05)
        result = _reproject(src, "EPSG:32632", src_transform, "EPSG:4326", dst_transform, (5, 5))
        assert result.shape == (5, 5)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        np.testing.assert_allclose(valid, 42.0, atol=1e-3)


class TestBilinearResample:
    def test_exact_pixel_centres(self):
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        # Pixel centres at 0.5, 1.5
        rows = np.array([[0.5, 0.5], [1.5, 1.5]])
        cols = np.array([[0.5, 1.5], [0.5, 1.5]])
        result = _bilinear_resample(src, rows, cols)
        np.testing.assert_array_almost_equal(result, src)

    def test_midpoint_interpolation(self):
        src = np.array([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32)
        # Centre between columns 0 and 1 (col=1.0 in pixel-grid space)
        rows = np.array([[0.5]])
        cols = np.array([[1.0]])
        result = _bilinear_resample(src, rows, cols)
        assert result[0, 0] == pytest.approx(1.0)

    def test_out_of_bounds_is_nan(self):
        src = np.ones((3, 3), dtype=np.float32)
        rows = np.array([[10.0]])
        cols = np.array([[10.0]])
        result = _bilinear_resample(src, rows, cols)
        assert np.isnan(result[0, 0])


# ---------------------------------------------------------------------------
# Resampling methods
# ---------------------------------------------------------------------------

_UNIT = Affine(1.0, 0, 0.0, 0, -1.0, 4.0)     # 1° pixels, 4×4 over (0..4, 0..4)
_HALF = Affine(2.0, 0, 0.0, 0, -2.0, 4.0)     # 2° pixels, 2×2 over the same area


def _align(array, src_transform, dst_transform, dst_shape, method, src_crs="EPSG:4326"):
    data = RasterData(array=np.asarray(array, dtype=np.float32), crs=src_crs, transform=src_transform)
    grid = GridSpec(crs="EPSG:4326", transform=dst_transform, shape=dst_shape)
    return align_to_grid(data, grid, resampling=method)


class TestResampling:
    block = np.array(
        [
            [1, 2, 5, 5],
            [3, 3, 5, 7],
            [0, 0, 9, 9],
            [0, 4, 9, 1],
        ],
        dtype=np.float32,
    )

    def test_default_is_bilinear(self):
        data = RasterData(array=self.block, crs="EPSG:4326", transform=_UNIT)
        grid = GridSpec(crs="EPSG:4326", transform=_HALF, shape=(2, 2))
        np.testing.assert_array_equal(
            align_to_grid(data, grid), align_to_grid(data, grid, resampling="bilinear")
        )

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="resampling must be one of"):
            _align(self.block, _UNIT, _HALF, (2, 2), "cubic")

    @pytest.mark.parametrize(
        "method, expected",
        [
            ("average", [[2.25, 5.5], [1.0, 7.0]]),
            ("min", [[1, 5], [0, 1]]),
            ("max", [[3, 7], [4, 9]]),
            ("mode", [[3, 5], [0, 9]]),
        ],
    )
    def test_downsampling_aggregates_each_block(self, method, expected):
        result = _align(self.block, _UNIT, _HALF, (2, 2), method)
        np.testing.assert_allclose(result, np.array(expected, dtype=np.float32))

    def test_max_keeps_a_single_pixel_spike(self):
        src = np.zeros((10, 10), dtype=np.float32)
        src[2, 2] = 80.0
        fine = Affine(0.1, 0, 0.0, 0, -0.1, 50.0)
        coarse = Affine(0.2, 0, 0.0, 0, -0.2, 50.0)
        assert _align(src, fine, coarse, (5, 5), "max")[1, 1] == pytest.approx(80.0)
        assert _align(src, fine, coarse, (5, 5), "bilinear")[1, 1] == pytest.approx(20.0)

    def test_nearest_produces_only_source_classes(self):
        classes = np.array([[1, 1, 2, 2]] * 4, dtype=np.float32)
        # Grid shifted by half a source pixel: bilinear would blend 1 and 2.
        shifted = Affine(1.0, 0, 0.5, 0, -1.0, 4.0)
        bilinear = _align(classes, _UNIT, shifted, (4, 3), "bilinear")
        nearest = _align(classes, _UNIT, shifted, (4, 3), "nearest")
        assert 1.5 in bilinear
        assert set(np.unique(nearest[~np.isnan(nearest)])) <= {1.0, 2.0}

    def test_mode_tie_goes_to_smallest_value(self):
        src = np.array([[4, 2], [2, 4]], dtype=np.float32)
        result = _align(src, Affine(1.0, 0, 0.0, 0, -1.0, 2.0),
                        Affine(2.0, 0, 0.0, 0, -2.0, 2.0), (1, 1), "mode")
        assert result[0, 0] == 2.0

    @pytest.mark.parametrize("method", ["mode", "average", "min", "max"])
    def test_aggregation_ignores_nan(self, method):
        src = self.block.copy()
        src[0, 0] = np.nan           # partly missing block
        src[2:, :2] = np.nan         # fully missing block
        result = _align(src, _UNIT, _HALF, (2, 2), method)
        expected_first = {"mode": 3.0, "average": 8 / 3, "min": 2.0, "max": 3.0}[method]
        assert result[0, 0] == pytest.approx(expected_first)
        assert np.isnan(result[1, 0])
        assert not np.isnan(result[0, 1])

    @pytest.mark.parametrize("method", ["nearest", "mode", "average", "min", "max"])
    def test_outside_extent_is_nan(self, method):
        # Destination 4×4 at 2° starting at the source origin: only the top-left 2×2 overlaps.
        wide = Affine(2.0, 0, 0.0, 0, -2.0, 4.0)
        result = _align(self.block, _UNIT, wide, (4, 4), method)
        assert not np.isnan(result[:2, :2]).any()
        assert np.isnan(result[2:, :]).all()
        assert np.isnan(result[:, 2:]).all()

    @pytest.mark.parametrize("method", ["mode", "average", "min", "max"])
    def test_upsampling_matches_nearest(self, method):
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        coarse = Affine(1.0, 0, 0.0, 0, -1.0, 2.0)
        fine = Affine(0.25, 0, 0.0, 0, -0.25, 2.0)
        np.testing.assert_array_equal(
            _align(src, coarse, fine, (8, 8), method),
            _align(src, coarse, fine, (8, 8), "nearest"),
        )

    @pytest.mark.parametrize("method", ["nearest", "mode", "average", "min", "max"])
    def test_grid_match_returns_source_unchanged(self, method):
        np.testing.assert_array_equal(
            _align(self.block, _UNIT, _UNIT, (4, 4), method), self.block
        )

    @pytest.mark.parametrize("method", ["nearest", "mode", "average", "min", "max"])
    def test_cross_crs_keeps_a_uniform_value(self, method):
        src = np.full((100, 100), 3.0, dtype=np.float32)
        src_transform = Affine(1000.0, 0, 450000.0, 0, -1000.0, 6701444.0)
        dst_transform = Affine(0.05, 0, 8.97, 0, -0.05, 60.05)
        result = _align(src, src_transform, dst_transform, (5, 5), method, src_crs="EPSG:32632")
        valid = result[~np.isnan(result)]
        assert valid.size > 0
        np.testing.assert_allclose(valid, 3.0)

    def test_cross_crs_max_finds_a_spike(self):
        """A one-pixel spike in UTM survives max onto a coarser lat/lon grid."""
        src = np.zeros((100, 100), dtype=np.float32)
        src[50, 50] = 99.0
        src_transform = Affine(100.0, 0, 495000.0, 0, -100.0, 6656444.0)
        transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
        lon0, lat1 = transformer.transform(495000.0, 6656444.0)
        lon1, lat0 = transformer.transform(505000.0, 6646444.0)
        res = 0.02
        dst_transform = Affine(res, 0, lon0, 0, -res, lat1)
        shape = (int((lat1 - lat0) / res), int((lon1 - lon0) / res))
        result = _align(src, src_transform, dst_transform, shape, "max", src_crs="EPSG:32632")
        assert np.nanmax(result) == pytest.approx(99.0)
        assert np.nanmax(_align(src, src_transform, dst_transform, shape, "bilinear",
                                src_crs="EPSG:32632")) < 99.0

    def test_aggregation_is_chunked(self, monkeypatch):
        """Results do not depend on how many rows are processed at once."""
        import geobn.grid as grid_module

        rng = np.random.default_rng(0)
        src = rng.integers(0, 5, (40, 40)).astype(np.float32)
        fine = Affine(0.1, 0, 0.0, 0, -0.1, 4.0)
        coarse = Affine(0.4, 0, 0.0, 0, -0.4, 4.0)
        whole = _align(src, fine, coarse, (10, 10), "mode")
        monkeypatch.setattr(grid_module, "_SAMPLE_BUDGET", 16)
        np.testing.assert_array_equal(_align(src, fine, coarse, (10, 10), "mode"), whole)
