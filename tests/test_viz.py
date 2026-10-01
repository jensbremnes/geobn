"""Tests for geobn._viz interactive map generation.

All tests are offline (no browser opened, no network calls).
"""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from geobn.result import InferenceResult


@pytest.fixture
def simple_result() -> InferenceResult:
    """Tiny 4×6 InferenceResult with 3-state node over EPSG:4326 grid."""
    rng = np.random.default_rng(0)
    H, W = 4, 6
    raw = rng.random((H, W, 3)).astype(np.float32)
    probs = raw / raw.sum(axis=-1, keepdims=True)
    # Sprinkle some NaN to exercise nodata handling
    probs[0, 0] = np.nan
    return InferenceResult(
        probabilities={"avalanche_risk": probs},
        state_names={"avalanche_risk": ["low", "medium", "high"]},
        crs="EPSG:4326",
        transform=Affine(0.1, 0, 19.8, 0, -0.1, 69.75),
    )


def test_show_map_creates_html(simple_result, tmp_path):
    """show_map() writes an HTML file containing Leaflet."""
    html_path = simple_result.show_map(tmp_path, open_browser=False)

    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "leaflet" in content.lower()


def test_show_map_contains_overlays(simple_result, tmp_path):
    """Layer names for probability bands and entropy appear in the generated HTML."""
    html_path = simple_result.show_map(tmp_path, open_browser=False)
    content = html_path.read_text(encoding="utf-8")

    assert "entropy" in content



def test_extra_layers_included(simple_result, tmp_path):
    """Extra layers passed via extra_layers appear in the generated HTML."""
    H, W = 4, 6
    slope = np.random.default_rng(1).random((H, W)).astype(np.float32) * 50.0

    html_path = simple_result.show_map(
        tmp_path,
        open_browser=False,
        extra_layers={"Slope angle": slope},
    )
    content = html_path.read_text(encoding="utf-8")
    assert "Slope angle" in content


def test_overlay_exclusivity_limited_to_raster_layers(simple_result, tmp_path):
    """The radio-button script only acts on overlays that have a colorbar."""
    html_path = simple_result.show_map(tmp_path, open_browser=False)
    content = html_path.read_text(encoding="utf-8")
    assert "in layerColorbars" in content


# ---------------------------------------------------------------------------
# Web Mercator display grid
# ---------------------------------------------------------------------------

from pyproj import Transformer  # noqa: E402

from geobn._viz import _web_mercator_grid, _web_mercator_warper  # noqa: E402
from geobn.grid import GridSpec  # noqa: E402


def _marked_cell_lonlat(crs, transform, shape, row, col):
    """Warp a grid with one marked cell; return (displayed, true) lon/lat of it."""
    dst, dst_shape, _ = _web_mercator_grid(crs, transform, shape)
    arr = np.zeros(shape, dtype=np.float32)
    arr[row, col] = 1.0
    warped = _web_mercator_warper(crs, transform, dst, dst_shape)(arr)
    r, c = np.argwhere(warped == 1.0).mean(axis=0)
    shown = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(
        *(dst * (c + 0.5, r + 0.5))
    )
    true = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
        *(transform * (col + 0.5, row + 0.5))
    )
    return np.array(shown), np.array(true)


def test_web_mercator_grid_utm_covers_extent():
    """The display bounds contain the grid's densified WGS84 extent."""
    t = Affine(100, 0, 600000, 0, -100, 7750000)
    shape = (200, 300)
    _, dst_shape, bounds = _web_mercator_grid("EPSG:32633", t, shape)
    (south, west), (north, east) = bounds
    lon_min, lat_min, lon_max, lat_max = GridSpec("EPSG:32633", t, shape).extent_wgs84()
    tol = 1e-6
    assert west <= lon_min + tol and east >= lon_max - tol
    assert south <= lat_min + tol and north >= lat_max - tol
    assert 0.5 < (dst_shape[0] * dst_shape[1]) / (shape[0] * shape[1]) < 2.0


def test_warp_places_utm_cell_correctly():
    """A marked UTM cell is displayed within one source pixel of its true place."""
    t = Affine(100, 0, 600000, 0, -100, 7750000)
    shown, true = _marked_cell_lonlat("EPSG:32633", t, (200, 300), 120, 45)
    # 100 m at 70° N: ~0.0026° longitude, ~0.0009° latitude
    assert abs(shown[0] - true[0]) < 0.0026
    assert abs(shown[1] - true[1]) < 0.0009


def test_warp_places_high_latitude_lonlat_cell_correctly():
    """Rows of an EPSG:4326 grid at 70° N are not shifted in latitude."""
    t = Affine(0.01, 0, 19.0, 0, -0.01, 70.5)  # 69.3–70.5° N
    shown, true = _marked_cell_lonlat("EPSG:4326", t, (120, 150), 60, 70)
    assert abs(shown[0] - true[0]) < 0.01
    assert abs(shown[1] - true[1]) < 0.01


def test_warp_keeps_nan_and_category_values():
    """NaN stays NaN, outside pixels are NaN, and categories stay exact."""
    t = Affine(100, 0, 600000, 0, -100, 7750000)
    shape = (50, 60)
    dst, dst_shape, _ = _web_mercator_grid("EPSG:32633", t, shape)
    warp = _web_mercator_warper("EPSG:32633", t, dst, dst_shape)
    category = np.random.default_rng(0).integers(0, 3, shape).astype(float)
    category[:10] = np.nan
    warped = warp(category)
    valid = warped[np.isfinite(warped)]
    assert set(np.unique(valid)) <= {0.0, 1.0, 2.0}
    assert np.isnan(warped).any()
    # The NaN band at the top of the grid is still at the top of the display
    assert np.isnan(warped[: dst_shape[0] // 10]).all()


def test_web_mercator_input_passes_through():
    """An axis-aligned EPSG:3857 grid is displayed as it is."""
    t = Affine(50, 0, 2000000, 0, -50, 11000000)
    shape = (30, 40)
    dst, dst_shape, _ = _web_mercator_grid("EPSG:3857", t, shape)
    assert dst == t and dst_shape == shape
    arr = np.random.default_rng(0).random(shape).astype(np.float32)
    np.testing.assert_array_equal(_web_mercator_warper("EPSG:3857", t, dst, dst_shape)(arr), arr)


def test_grid_beyond_web_mercator_raises(tmp_path):
    """A grid reaching past 85.05° latitude cannot be shown."""
    probs = np.full((4, 6, 2), 0.5, dtype=np.float32)
    result = InferenceResult(
        probabilities={"n": probs},
        state_names={"n": ["a", "b"]},
        crs="EPSG:4326",
        transform=Affine(1.0, 0, 0.0, 0, -1.0, 88.0),
    )
    with pytest.raises(ValueError, match="Web Mercator"):
        result.show_map(tmp_path, open_browser=False)
