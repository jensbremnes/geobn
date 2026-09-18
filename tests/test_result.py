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


# ---------------------------------------------------------------------------
# Summary layers
# ---------------------------------------------------------------------------

@pytest.fixture
def known_result():
    """2×2 grid, states low/medium/high, one NoData pixel."""
    probs = np.array(
        [
            [[0.2, 0.5, 0.3], [1.0, 0.0, 0.0]],
            [[0.1, 0.1, 0.8], [np.nan, np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    return InferenceResult(
        probabilities={"risk": probs},
        state_names={"risk": ["low", "medium", "high"]},
        crs="EPSG:4326",
        transform=Affine(0.1, 0, 0.0, 0, -0.1, 50.0),
    )


def _assert_layer(actual, expected):
    assert actual.shape == (2, 2)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, np.array(expected, dtype=np.float32), atol=1e-5)


def test_expected_value(known_result):
    ev = known_result.expected_value("risk", [10, 50, 90])
    _assert_layer(ev, [[0.2 * 10 + 0.5 * 50 + 0.3 * 90, 10.0], [0.1 * 10 + 0.1 * 50 + 0.8 * 90, np.nan]])


def test_expected_value_dict_matches_sequence(known_result):
    as_list = known_result.expected_value("risk", [10, 50, 90])
    as_dict = known_result.expected_value("risk", {"high": 90, "low": 10, "medium": 50})
    np.testing.assert_array_equal(as_list, as_dict)


def test_std(known_result):
    mean = 0.2 * 10 + 0.5 * 50 + 0.3 * 90
    var = 0.2 * (10 - mean) ** 2 + 0.5 * (50 - mean) ** 2 + 0.3 * (90 - mean) ** 2
    sd = known_result.std("risk", [10, 50, 90])
    assert sd[0, 0] == pytest.approx(np.sqrt(var), rel=1e-5)
    assert sd[0, 1] == pytest.approx(0.0, abs=1e-5)  # certain pixel
    assert np.isnan(sd[1, 1])


@pytest.mark.parametrize("values, match", [
    ([10, 50], "3 entries"),
    ({"low": 10, "medium": 50}, "missing"),
    ({"low": 10, "medium": 50, "high": 90, "extreme": 99}, "unknown"),
    ([10, np.nan, 90], "finite"),
])
def test_state_values_validation(known_result, values, match):
    with pytest.raises(ValueError, match=match):
        known_result.expected_value("risk", values)


def test_unknown_node_raises(known_result):
    with pytest.raises(KeyError, match="available"):
        known_result.mode("nope")


def test_mode_and_mode_probability(known_result):
    _assert_layer(known_result.mode("risk"), [[1, 0], [2, np.nan]])
    _assert_layer(known_result.mode_probability("risk"), [[0.5, 1.0], [0.8, np.nan]])


def test_exceedance(known_result):
    _assert_layer(known_result.exceedance("risk", "medium"), [[0.8, 0.0], [0.9, np.nan]])
    _assert_layer(known_result.exceedance("risk", 2), [[0.3, 0.0], [0.8, np.nan]])
    _assert_layer(known_result.exceedance("risk", "low"), [[1.0, 1.0], [1.0, np.nan]])


@pytest.mark.parametrize("state, exc", [("extreme", ValueError), (3, ValueError), (-1, ValueError), (1.0, TypeError)])
def test_exceedance_bad_state(known_result, state, exc):
    with pytest.raises(exc):
        known_result.exceedance("risk", state)


def test_ignorance(known_result):
    _assert_layer(known_result.ignorance("risk", 0.6), [[np.nan, 0], [2, np.nan]])
    _assert_layer(known_result.ignorance("risk", 0.5), [[1, 0], [2, np.nan]])
    with pytest.raises(ValueError, match="threshold"):
        known_result.ignorance("risk", 0.0)


def test_quantile(known_result):
    # Cumulative: [0.2, 0.7, 1.0], [1, 1, 1], [0.1, 0.2, 1.0]
    _assert_layer(known_result.quantile("risk", 0.5), [[1, 0], [2, np.nan]])
    _assert_layer(known_result.quantile("risk", 0.2), [[0, 0], [1, np.nan]])
    _assert_layer(known_result.quantile("risk", 0.95), [[2, 0], [2, np.nan]])


def test_quantile_one_is_highest_possible_state():
    probs = np.array([[[0.5, 0.5, 0.0]]], dtype=np.float32)
    result = InferenceResult({"r": probs}, {"r": ["a", "b", "c"]}, "EPSG:4326", Affine.identity())
    assert result.quantile("r", 1.0)[0, 0] == 1


@pytest.mark.parametrize("q", [0.0, -0.1, 1.5])
def test_quantile_bad_q(known_result, q):
    with pytest.raises(ValueError, match="q must be"):
        known_result.quantile("risk", q)


# ---------------------------------------------------------------------------
# Exporting extra layers
# ---------------------------------------------------------------------------

def test_to_geotiff_band_descriptions(known_result, tmp_path):
    import rasterio

    known_result.to_geotiff(tmp_path)
    with rasterio.open(tmp_path / "risk.tif") as src:
        assert src.descriptions == ("low", "medium", "high", "entropy")


def test_to_geotiff_extra_layers(known_result, tmp_path):
    import rasterio

    score = known_result.expected_value("risk", [10, 50, 90])
    known_result.to_geotiff(tmp_path, layers={"risk_score": score})
    with rasterio.open(tmp_path / "risk_score.tif") as src:
        assert src.count == 1
        assert src.descriptions == ("risk_score",)
        assert src.transform == known_result.transform
        np.testing.assert_allclose(src.read(1), score, equal_nan=True)


def test_to_xarray_extra_layers(known_result):
    score = known_result.expected_value("risk", [10, 50, 90])
    ds = known_result.to_xarray(layers={"risk_score": score})
    assert ds["risk_score"].dims == ("y", "x")
    np.testing.assert_allclose(ds["risk_score"].values, score, equal_nan=True)


@pytest.mark.parametrize("layers, match", [
    ({"bad": np.zeros((3, 3))}, "shape"),
    ({"risk": np.zeros((2, 2))}, "clashes"),
    ({"a/b": np.zeros((2, 2))}, "separators"),
    ({"": np.zeros((2, 2))}, "separators"),
])
def test_extra_layer_validation(known_result, tmp_path, layers, match):
    with pytest.raises(ValueError, match=match):
        known_result.to_geotiff(tmp_path, layers=layers)
    with pytest.raises(ValueError, match=match):
        known_result.to_xarray(layers=layers)


def test_to_xarray_layer_cannot_shadow_entropy(known_result):
    with pytest.raises(ValueError, match="clashes"):
        known_result.to_xarray(layers={"risk_entropy": np.zeros((2, 2))})
