"""Tests for InferenceResult."""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from geobn.result import InferenceResult


@pytest.fixture
def sample_result():
    H, W = 4, 4
    probs = np.random.default_rng(0).dirichlet([1, 1, 1], size=(H, W)).astype(np.float32)
    return InferenceResult(
        probabilities={"fire_risk": probs},
        state_names={"fire_risk": ["low", "medium", "high"]},
        crs="EPSG:4326",
        transform=Affine(0.1, 0, 0.0, 0, -0.1, 50.0),
    )


def test_entropy_shape(sample_result):
    ent = sample_result.entropy("fire_risk")
    assert ent.shape == (4, 4)
    assert np.all(ent >= 0)


def test_to_geotiff_creates_file(sample_result, tmp_path):
    sample_result.to_geotiff(tmp_path)
    out = tmp_path / "fire_risk.tif"
    assert out.exists()

    import rasterio

    with rasterio.open(out) as src:
        # 3 probability bands + 1 entropy band
        assert src.count == 4
        assert src.height == 4
        assert src.width == 4


def test_to_xarray_dims(sample_result):
    ds = sample_result.to_xarray()
    assert "fire_risk" in ds
    assert "fire_risk_entropy" in ds
    da = ds["fire_risk"]
    assert set(da.dims) == {"state", "y", "x"}
    assert list(da.coords["state"].values) == ["low", "medium", "high"]
    assert da.shape == (3, 4, 4)
