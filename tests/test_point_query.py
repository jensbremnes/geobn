"""Tests for query_point / query_batch — point and trajectory lookups on the
precomputed conditional table."""

from __future__ import annotations

import numpy as np
import pytest
from pgmpy.inference import VariableElimination

import geobn


@pytest.fixture
def point_bn(fire_risk_model) -> geobn.GeoBayesianNetwork:
    """fire_risk BN wired for point queries (sources are irrelevant here —
    the registered inputs only define the evidence-node set)."""
    bn = geobn.GeoBayesianNetwork(fire_risk_model)
    bn.set_input("slope", geobn.ConstantSource(10.0))
    bn.set_input("rainfall", geobn.ConstantSource(50.0))
    bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
    bn.set_discretization("rainfall", [0, 30, 100, 300], ["low", "medium", "high"])
    bn.precompute(query=["fire_risk"])
    return bn


def test_query_point_matches_variable_elimination(point_bn, fire_risk_model):
    ve = VariableElimination(fire_risk_model)
    for slope_state in ("flat", "moderate", "steep"):
        for rain_state in ("low", "medium", "high"):
            expected = ve.query(
                ["fire_risk"],
                evidence={"slope": slope_state, "rainfall": rain_state},
                show_progress=False,
            )
            got = point_bn.query_point({"slope": slope_state, "rainfall": rain_state})
            for i, state in enumerate(expected.state_names["fire_risk"]):
                assert got["fire_risk"][state] == pytest.approx(
                    float(expected.values[i]), abs=1e-6
                )


def test_query_point_discretizes_continuous_evidence(point_bn):
    # 35 degrees -> "steep", 20 mm -> "low"
    by_value = point_bn.query_point({"slope": 35.0, "rainfall": 20.0})
    by_state = point_bn.query_point({"slope": "steep", "rainfall": "low"})
    assert by_value == by_state


def test_query_point_probabilities_sum_to_one(point_bn):
    out = point_bn.query_point({"slope": 5.0, "rainfall": 150.0})
    assert sum(out["fire_risk"].values()) == pytest.approx(1.0, abs=1e-5)


def test_query_point_requires_precompute(fire_risk_model):
    bn = geobn.GeoBayesianNetwork(fire_risk_model)
    bn.set_input("slope", geobn.ConstantSource(10.0))
    with pytest.raises(RuntimeError, match="precompute"):
        bn.query_point({"slope": "flat"})


def test_query_point_missing_evidence(point_bn):
    with pytest.raises(ValueError, match="rainfall"):
        point_bn.query_point({"slope": "flat"})


def test_query_point_unknown_state(point_bn):
    with pytest.raises(ValueError, match="vertical"):
        point_bn.query_point({"slope": "vertical", "rainfall": "low"})


def test_query_point_nan_evidence_raises(point_bn):
    with pytest.raises(ValueError, match="NaN"):
        point_bn.query_point({"slope": float("nan"), "rainfall": "low"})


def test_query_point_unknown_query_node(point_bn):
    with pytest.raises(ValueError, match="not precomputed"):
        point_bn.query_point({"slope": "flat", "rainfall": "low"}, query=["slope"])


def test_query_batch_matches_point(point_bn):
    slope = np.array([5.0, 20.0, 45.0, 12.0])
    rain = np.array([10.0, 60.0, 250.0, 40.0])
    batch = point_bn.query_batch({"slope": slope, "rainfall": rain})["fire_risk"]
    assert batch.shape == (4, 3)
    for k in range(4):
        point = point_bn.query_point({"slope": float(slope[k]), "rainfall": float(rain[k])})
        np.testing.assert_allclose(
            batch[k], list(point["fire_risk"].values()), atol=1e-6
        )


def test_query_batch_broadcasts_scalars(point_bn):
    slope = np.array([5.0, 20.0, 45.0])
    batch = point_bn.query_batch({"slope": slope, "rainfall": "high"})["fire_risk"]
    assert batch.shape == (3, 3)
    for k, s in enumerate(slope):
        point = point_bn.query_point({"slope": float(s), "rainfall": "high"})
        np.testing.assert_allclose(batch[k], list(point["fire_risk"].values()), atol=1e-6)


def test_query_batch_nan_propagates_as_nan_row(point_bn):
    slope = np.array([5.0, np.nan, 45.0])
    rain = np.array([10.0, 60.0, 250.0])
    batch = point_bn.query_batch({"slope": slope, "rainfall": rain})["fire_risk"]
    assert np.all(np.isnan(batch[1]))
    assert np.all(np.isfinite(batch[[0, 2]]))


def test_query_batch_mismatched_lengths(point_bn):
    with pytest.raises(ValueError, match="mismatched"):
        point_bn.query_batch(
            {"slope": np.array([1.0, 2.0]), "rainfall": np.array([1.0, 2.0, 3.0])}
        )


def test_query_point_after_save_load_roundtrip(point_bn, fire_risk_model, tmp_path):
    """The deployment path: table built offline, shipped, queried onboard."""
    path = tmp_path / "table.npz"
    point_bn.save_precomputed(path)

    bn = geobn.GeoBayesianNetwork(fire_risk_model)
    bn.set_input("slope", geobn.ConstantSource(10.0))
    bn.set_input("rainfall", geobn.ConstantSource(50.0))
    bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
    bn.set_discretization("rainfall", [0, 30, 100, 300], ["low", "medium", "high"])
    bn.load_precomputed(path)

    expected = point_bn.query_point({"slope": 35.0, "rainfall": 20.0})
    assert bn.query_point({"slope": 35.0, "rainfall": 20.0}) == expected
