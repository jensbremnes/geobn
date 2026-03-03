"""Tests for geobn._viz interactive map generation.

All tests are offline (no browser opened, no network calls).
The entire module is skipped when folium is not installed.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest
from affine import Affine

folium = pytest.importorskip("folium")

from geobn.result import InferenceResult  # noqa: E402


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
    """Layer names for P(high) and entropy appear in the generated HTML."""
    html_path = simple_result.show_map(tmp_path, open_browser=False)
    content = html_path.read_text(encoding="utf-8")

    assert "P(high)" in content
    assert "entropy" in content


def test_show_map_skips_without_folium(simple_result, tmp_path):
    """ImportError with a helpful message is raised when folium is absent."""
    with patch.dict(
        sys.modules,
        {"folium": None, "branca": None, "branca.colormap": None},
    ):
        with pytest.raises(ImportError, match="folium"):
            simple_result.show_map(tmp_path, open_browser=False)


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
