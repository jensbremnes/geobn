"""Tests for data source classes."""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from affine import Affine

import geobn
from geobn._types import RasterData
from geobn.grid import GridSpec
from geobn.sources.point_grid_source import PointGridSource
from geobn.sources.wcs_source import WCSSource


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid() -> GridSpec:
    """5×5 grid over a small area in WGS84."""
    transform = Affine(0.1, 0, 5.0, 0, -0.1, 62.0)
    return GridSpec(crs="EPSG:4326", transform=transform, shape=(5, 5))


def _make_geotiff_bytes(array: np.ndarray) -> bytes:
    """Build minimal in-memory GeoTIFF bytes using rasterio."""
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    H, W = array.shape
    transform = from_bounds(5.0, 60.0, 6.0, 62.0, W, H)
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=H,
            width=W,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(array.astype("float32"), 1)
        return memfile.read()


# ---------------------------------------------------------------------------
# ArraySource
# ---------------------------------------------------------------------------

def test_array_source_roundtrip(slope_array, reference_transform):
    source = geobn.ArraySource(slope_array, crs="EPSG:4326", transform=reference_transform)
    data = source.fetch()
    assert isinstance(data, RasterData)
    np.testing.assert_array_equal(data.array, slope_array)
    assert data.crs == "EPSG:4326"
    assert data.transform == reference_transform


def test_array_source_converts_to_float32(reference_transform):
    arr = np.ones((5, 5), dtype=np.float64)
    source = geobn.ArraySource(arr, crs="EPSG:4326", transform=reference_transform)
    assert source.fetch().array.dtype == np.float32


def test_array_source_rejects_3d(reference_transform):
    arr = np.ones((5, 5, 3))
    with pytest.raises(ValueError, match="2-D"):
        geobn.ArraySource(arr, crs="EPSG:4326", transform=reference_transform)


def test_array_source_no_crs_returns_none_metadata():
    """ArraySource with no crs/transform → RasterData with crs=None, transform=None."""
    arr = np.ones((5, 5), dtype=np.float32)
    source = geobn.ArraySource(arr)
    data = source.fetch()
    assert isinstance(data, RasterData)
    assert data.crs is None
    assert data.transform is None
    np.testing.assert_array_equal(data.array, arr)


# ---------------------------------------------------------------------------
# ConstantSource
# ---------------------------------------------------------------------------

def test_constant_source_returns_scalar():
    source = geobn.ConstantSource(0.6)
    data = source.fetch()
    assert data.crs is None
    assert data.transform is None
    assert data.array.shape == (1, 1)
    assert float(data.array[0, 0]) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# RasterSource
# ---------------------------------------------------------------------------

def test_raster_source_missing_file_raises(tmp_path):
    """RasterSource.fetch() should raise an error for a non-existent file."""
    source = geobn.RasterSource(tmp_path / "nonexistent.tif")
    with pytest.raises(Exception):
        source.fetch()


# ---------------------------------------------------------------------------
# PointGridSource
# ---------------------------------------------------------------------------

def test_point_grid_source_requires_grid():
    source = PointGridSource(fn=lambda lat, lon: 42.0, sample_points=3)
    with pytest.raises(ValueError, match="grid context"):
        source.fetch(grid=None)


def test_point_grid_source_returns_correct_shape(small_grid):
    n = 3
    source = PointGridSource(fn=lambda lat, lon: lat + lon, sample_points=n, delay=0.0)
    data = source.fetch(grid=small_grid)
    assert data.array.shape == (n, n)
    assert data.crs == "EPSG:4326"
    assert data.transform is not None


def test_point_grid_source_nan_for_none_return(small_grid):
    source = PointGridSource(fn=lambda lat, lon: None, sample_points=2, delay=0.0)
    data = source.fetch(grid=small_grid)
    assert np.all(np.isnan(data.array))


def test_point_grid_source_nan_return(small_grid):
    source = PointGridSource(fn=lambda lat, lon: float("nan"), sample_points=2, delay=0.0)
    data = source.fetch(grid=small_grid)
    assert np.all(np.isnan(data.array))


def test_point_grid_source_single_point_broadcasts(small_grid):
    """sample_points=1 should return a 1×1 raster with crs=None (like ConstantSource)."""
    source = PointGridSource(fn=lambda lat, lon: 7.5, sample_points=1, delay=0.0)
    data = source.fetch(grid=small_grid)
    assert data.array.shape == (1, 1)
    assert data.crs is None
    assert data.transform is None
    assert float(data.array[0, 0]) == pytest.approx(7.5)


def test_point_grid_source_affine_half_pixel_offset(small_grid):
    """Affine transform should extend half a pixel beyond the outermost samples."""
    n = 3
    source = PointGridSource(fn=lambda lat, lon: 1.0, sample_points=n, delay=0.0)
    data = source.fetch(grid=small_grid)
    lon_min, lat_min, lon_max, lat_max = small_grid.extent_wgs84()
    pixel_w = (lon_max - lon_min) / (n - 1)
    pixel_h = (lat_max - lat_min) / (n - 1)
    # Top-left corner should be half a pixel outside the first sample
    assert data.transform.c == pytest.approx(lon_min - pixel_w / 2)
    assert data.transform.f == pytest.approx(lat_max + pixel_h / 2)


# ---------------------------------------------------------------------------
# WCSSource — valid_range masking
# ---------------------------------------------------------------------------

def test_wcs_source_valid_range_masks_sentinels(small_grid):
    """valid_range should replace out-of-range values with NaN."""
    # Build a GeoTIFF with values including out-of-range sentinels
    raw = np.array([[100.0, -9999.0], [50.0, 9001.0]], dtype=np.float32)
    tiff_bytes = _make_geotiff_bytes(raw)

    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.content = tiff_bytes
    mock_response.elapsed.total_seconds.return_value = 0.1

    source = WCSSource(
        url="https://example.com/wcs",
        layer="test_layer",
        version="1.0.0",
        valid_range=(-500.0, 9000.0),
    )

    with patch("requests.get", return_value=mock_response):
        data = source.fetch(grid=small_grid)

    assert data.array.shape == raw.shape
    # 100 and 50 are within range; -9999 and 9001 should be NaN
    assert np.isfinite(data.array[0, 0])   # 100
    assert np.isnan(data.array[0, 1])      # -9999
    assert np.isfinite(data.array[1, 0])   # 50
    assert np.isnan(data.array[1, 1])      # 9001


def test_wcs_source_no_valid_range_passes_through(small_grid):
    """Without valid_range, out-of-range values are returned as-is."""
    raw = np.array([[100.0, -9999.0]], dtype=np.float32)
    tiff_bytes = _make_geotiff_bytes(raw)

    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.content = tiff_bytes
    mock_response.elapsed.total_seconds.return_value = 0.1

    source = WCSSource(
        url="https://example.com/wcs",
        layer="test_layer",
        version="1.0.0",
    )

    with patch("requests.get", return_value=mock_response):
        data = source.fetch(grid=small_grid)

    assert np.isfinite(data.array[0, 0])   # 100 — intact
    assert data.array[0, 1] == pytest.approx(-9999.0)  # not masked
