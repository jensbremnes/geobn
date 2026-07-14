"""Tests for geobn.analysis — tornado sensitivity and value of information."""

from __future__ import annotations

import numpy as np
import pytest
from pgmpy.inference import VariableElimination

import geobn
from geobn.analysis import tornado


@pytest.fixture
def baseline() -> dict[str, str]:
    return {"slope": "steep", "rainfall": "low"}


def test_tornado_swing_hand_computed(fire_risk_model, baseline):
    """Swing must equal max-min of the directly computed posteriors."""
    ve = VariableElimination(fire_risk_model)
    entries = {e.node: e for e in tornado(fire_risk_model, "fire_risk", "high", baseline)}

    high = list(
        fire_risk_model.get_cpds("fire_risk").state_names["fire_risk"]
    ).index("high")
    p = {
        s: float(
            ve.query(
                ["fire_risk"], evidence={"slope": s, "rainfall": "low"}, show_progress=False
            ).values[high]
        )
        for s in ("flat", "moderate", "steep")
    }
    assert entries["slope"].swing == pytest.approx(max(p.values()) - min(p.values()), abs=1e-9)
    assert entries["slope"].p_by_state == pytest.approx(p, abs=1e-9)
    assert entries["slope"].baseline_state == "steep"


def test_tornado_sorted_by_swing(fire_risk_model, baseline):
    entries = tornado(fire_risk_model, "fire_risk", "high", baseline)
    swings = [e.swing for e in entries]
    assert swings == sorted(swings, reverse=True)
    assert {e.node for e in entries} == set(baseline)


def test_voi_nonnegative_and_zero_for_irrelevant_node(fire_risk_model, baseline):
    entries = {e.node: e for e in tornado(fire_risk_model, "fire_risk", "high", baseline)}
    assert all(e.voi_bits >= 0.0 for e in entries.values())
    # both parents influence fire_risk in this model, so VOI is strictly positive
    assert entries["slope"].voi_bits > 0.0
    assert entries["rainfall"].voi_bits > 0.0


def test_voi_hand_computed(fire_risk_model, baseline):
    """VOI(slope) = H(target | rest) - E_slope[H(target | rest, slope)]."""
    ve = VariableElimination(fire_risk_model)

    def entropy(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    rest = {"rainfall": "low"}
    h_prior = entropy(ve.query(["fire_risk"], evidence=rest, show_progress=False).values)
    predictive = ve.query(["slope"], evidence=rest, show_progress=False).values
    states = list(fire_risk_model.get_cpds("slope").state_names["slope"])
    h_expected = sum(
        w
        * entropy(
            ve.query(
                ["fire_risk"], evidence={**rest, "slope": s}, show_progress=False
            ).values
        )
        for w, s in zip(predictive, states)
    )
    expected_voi = max(h_prior - h_expected, 0.0)

    entries = {e.node: e for e in tornado(fire_risk_model, "fire_risk", "high", baseline)}
    assert entries["slope"].voi_bits == pytest.approx(expected_voi, abs=1e-9)


def test_accepts_geo_bayesian_network(fire_risk_model, baseline):
    bn = geobn.GeoBayesianNetwork(fire_risk_model)
    from_wrapper = tornado(bn, "fire_risk", "high", baseline)
    from_model = tornado(fire_risk_model, "fire_risk", "high", baseline)
    assert [(e.node, e.swing) for e in from_wrapper] == [
        (e.node, e.swing) for e in from_model
    ]


def test_validation_errors(fire_risk_model, baseline):
    with pytest.raises(ValueError, match="Unknown target"):
        tornado(fire_risk_model, "nope", "high", baseline)
    with pytest.raises(ValueError, match="Unknown state 'extreme'"):
        tornado(fire_risk_model, "fire_risk", "extreme", baseline)
    with pytest.raises(ValueError, match="Unknown baseline node"):
        tornado(fire_risk_model, "fire_risk", "high", {"wind": "low"})
    with pytest.raises(ValueError, match="cannot be part of the baseline"):
        tornado(
            fire_risk_model, "fire_risk", "high", {"fire_risk": "low", "slope": "flat"}
        )
    with pytest.raises(TypeError, match="Expected GeoBayesianNetwork"):
        tornado("not a model", "fire_risk", "high", baseline)
