"""Tests for the inference engine."""
from __future__ import annotations

import numpy as np
import pytest

import geobn.inference as inference_module
from geobn.inference import build_conditional_table, run_inference, shannon_entropy


class TestRunInference:
    def test_single_pixel_known_evidence(self, fire_risk_model):
        """For uniform 'flat' slope and 'low' rainfall, fire risk should be mostly low."""
        H, W = 2, 2
        evidence_state_grids = {
            "slope": np.zeros((H, W), dtype=np.int16),     # all "flat"
            "rainfall": np.zeros((H, W), dtype=np.int16),  # all "low"
        }
        evidence_state_names = {
            "slope": ["flat", "moderate", "steep"],
            "rainfall": ["low", "medium", "high"],
        }
        nodata_mask = np.zeros((H, W), dtype=bool)

        result = run_inference(
            model=fire_risk_model,
            evidence_state_grids=evidence_state_grids,
            evidence_state_names=evidence_state_names,
            query_nodes=["fire_risk"],
            query_state_names={"fire_risk": ["low", "medium", "high"]},
            nodata_mask=nodata_mask,
        )

        probs = result["fire_risk"]
        assert probs.shape == (H, W, 3)
        # P(low | flat, low) should be 0.7 per CPT
        assert probs[0, 0, 0] == pytest.approx(0.70, abs=1e-4)
        # Probabilities must sum to 1
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)

    def test_nodata_propagates_nan(self, fire_risk_model):
        H, W = 3, 3
        evidence_state_grids = {
            "slope": np.zeros((H, W), dtype=np.int16),
            "rainfall": np.zeros((H, W), dtype=np.int16),
        }
        nodata_mask = np.zeros((H, W), dtype=bool)
        nodata_mask[1, 1] = True  # one NoData pixel

        result = run_inference(
            model=fire_risk_model,
            evidence_state_grids=evidence_state_grids,
            evidence_state_names={
                "slope": ["flat", "moderate", "steep"],
                "rainfall": ["low", "medium", "high"],
            },
            query_nodes=["fire_risk"],
            query_state_names={"fire_risk": ["low", "medium", "high"]},
            nodata_mask=nodata_mask,
        )

        probs = result["fire_risk"]
        assert np.all(np.isnan(probs[1, 1, :]))
        assert not np.any(np.isnan(probs[0, 0, :]))

    def test_single_pixel_grid(self, fire_risk_model):
        """1×1 grid should produce a valid (1, 1, n_states) probability array."""
        H, W = 1, 1
        evidence_state_grids = {
            "slope": np.zeros((H, W), dtype=np.int16),
            "rainfall": np.zeros((H, W), dtype=np.int16),
        }
        nodata_mask = np.zeros((H, W), dtype=bool)
        result = run_inference(
            model=fire_risk_model,
            evidence_state_grids=evidence_state_grids,
            evidence_state_names={
                "slope": ["flat", "moderate", "steep"],
                "rainfall": ["low", "medium", "high"],
            },
            query_nodes=["fire_risk"],
            query_state_names={"fire_risk": ["low", "medium", "high"]},
            nodata_mask=nodata_mask,
        )
        probs = result["fire_risk"]
        assert probs.shape == (1, 1, 3)
        assert not np.any(np.isnan(probs))
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)

    def test_all_nodata_returns_nan_array(self, fire_risk_model):
        H, W = 2, 2
        evidence_state_grids = {
            "slope": np.zeros((H, W), dtype=np.int16),
            "rainfall": np.zeros((H, W), dtype=np.int16),
        }
        nodata_mask = np.ones((H, W), dtype=bool)

        result = run_inference(
            model=fire_risk_model,
            evidence_state_grids=evidence_state_grids,
            evidence_state_names={
                "slope": ["flat", "moderate", "steep"],
                "rainfall": ["low", "medium", "high"],
            },
            query_nodes=["fire_risk"],
            query_state_names={"fire_risk": ["low", "medium", "high"]},
            nodata_mask=nodata_mask,
        )
        assert np.all(np.isnan(result["fire_risk"]))

    def test_unique_combo_deduplication(self, fire_risk_model):
        """All pixels identical → inference runs once, all pixels get same result."""
        H, W = 5, 5
        evidence_state_grids = {
            "slope": np.full((H, W), 2, dtype=np.int16),    # all "steep"
            "rainfall": np.full((H, W), 2, dtype=np.int16),  # all "high"
        }
        nodata_mask = np.zeros((H, W), dtype=bool)
        result = run_inference(
            model=fire_risk_model,
            evidence_state_grids=evidence_state_grids,
            evidence_state_names={
                "slope": ["flat", "moderate", "steep"],
                "rainfall": ["low", "medium", "high"],
            },
            query_nodes=["fire_risk"],
            query_state_names={"fire_risk": ["low", "medium", "high"]},
            nodata_mask=nodata_mask,
        )
        probs = result["fire_risk"]
        # All pixels should have identical probabilities
        for i in range(H):
            for j in range(W):
                np.testing.assert_array_almost_equal(probs[i, j], probs[0, 0])


class TestBuildConditionalTable:
    def test_matches_per_combo_queries(self, fire_risk_model):
        """The single-joint-query table must equal one VE query per combination."""
        from pgmpy.inference import VariableElimination

        tables = build_conditional_table(
            fire_risk_model, ["slope", "rainfall"], ["fire_risk"]
        )
        assert tables is not None
        table = tables["fire_risk"]
        assert table.shape == (3, 3, 3)
        np.testing.assert_allclose(table.sum(axis=-1), 1.0, atol=1e-5)

        ve = VariableElimination(fire_risk_model)
        slope_states = ["flat", "moderate", "steep"]
        rainfall_states = ["low", "medium", "high"]
        for i, s in enumerate(slope_states):
            for j, r in enumerate(rainfall_states):
                expected = ve.query(
                    ["fire_risk"],
                    evidence={"slope": s, "rainfall": r},
                    show_progress=False,
                ).values
                np.testing.assert_allclose(table[i, j], expected, atol=1e-6)

    def test_axis_order_follows_evidence_nodes(self, fire_risk_model):
        """Swapping the evidence node order must transpose the table axes."""
        t_sr = build_conditional_table(fire_risk_model, ["slope", "rainfall"], ["fire_risk"])
        t_rs = build_conditional_table(fire_risk_model, ["rainfall", "slope"], ["fire_risk"])
        np.testing.assert_allclose(
            t_sr["fire_risk"], np.transpose(t_rs["fire_risk"], (1, 0, 2)), atol=1e-6
        )

    def test_size_bound_returns_none(self, fire_risk_model):
        result = build_conditional_table(
            fire_risk_model, ["slope", "rainfall"], ["fire_risk"], max_table_cells=1
        )
        assert result is None

    def test_query_node_in_evidence_returns_none(self, fire_risk_model):
        result = build_conditional_table(fire_risk_model, ["slope"], ["slope"])
        assert result is None


class TestStrategyEquivalence:
    @staticmethod
    def _random_inference_kwargs(model, seed=0, H=20, W=20):
        rng = np.random.default_rng(seed)
        evidence_state_grids = {
            "slope": rng.integers(0, 3, (H, W)).astype(np.int16),
            "rainfall": rng.integers(0, 3, (H, W)).astype(np.int16),
        }
        nodata_mask = np.zeros((H, W), dtype=bool)
        nodata_mask[0, 0] = True
        evidence_state_grids["slope"][0, 0] = -1
        return dict(
            model=model,
            evidence_state_grids=evidence_state_grids,
            evidence_state_names={
                "slope": ["flat", "moderate", "steep"],
                "rainfall": ["low", "medium", "high"],
            },
            query_nodes=["fire_risk"],
            query_state_names={"fire_risk": ["low", "medium", "high"]},
            nodata_mask=nodata_mask,
        )

    def test_joint_path_matches_loop_path(self, fire_risk_model, monkeypatch):
        kwargs = self._random_inference_kwargs(fire_risk_model)

        monkeypatch.setattr(inference_module, "_COMBO_LOOP_THRESHOLD", 10**9)
        loop_result = run_inference(**kwargs)

        monkeypatch.setattr(inference_module, "_COMBO_LOOP_THRESHOLD", 0)
        joint_result = run_inference(**kwargs)

        np.testing.assert_allclose(
            loop_result["fire_risk"], joint_result["fire_risk"],
            atol=1e-6, equal_nan=True,
        )

    def test_loop_fallback_when_table_rejected(self, fire_risk_model, monkeypatch):
        """If the table builder declines, the per-combo loop must still run."""
        kwargs = self._random_inference_kwargs(fire_risk_model)
        monkeypatch.setattr(inference_module, "_COMBO_LOOP_THRESHOLD", 0)
        monkeypatch.setattr(
            inference_module, "build_conditional_table", lambda *a, **k: None
        )
        result = run_inference(**kwargs)
        probs = result["fire_risk"]
        assert np.all(np.isnan(probs[0, 0, :]))
        valid = ~kwargs["nodata_mask"]
        np.testing.assert_allclose(probs[valid].sum(axis=-1), 1.0, atol=1e-5)

    def test_multiple_query_nodes_shared_elimination(self, fire_risk_model):
        """Evidence on slope only; query both fire_risk and rainfall."""
        H, W = 2, 2
        result = run_inference(
            model=fire_risk_model,
            evidence_state_grids={"slope": np.zeros((H, W), dtype=np.int16)},
            evidence_state_names={"slope": ["flat", "moderate", "steep"]},
            query_nodes=["fire_risk", "rainfall"],
            query_state_names={
                "fire_risk": ["low", "medium", "high"],
                "rainfall": ["low", "medium", "high"],
            },
            nodata_mask=np.zeros((H, W), dtype=bool),
        )
        # rainfall is independent of slope → its prior
        np.testing.assert_allclose(result["rainfall"][0, 0], [0.3, 0.4, 0.3], atol=1e-5)
        # P(fire_risk | slope=flat) marginalised over the rainfall prior
        np.testing.assert_allclose(result["fire_risk"][0, 0], [0.57, 0.30, 0.13], atol=1e-5)


class TestShannonEntropy:
    def test_uniform_distribution_max_entropy(self):
        probs = np.array([[[1 / 3, 1 / 3, 1 / 3]]])
        ent = shannon_entropy(probs)
        assert ent[0, 0] == pytest.approx(np.log2(3), rel=1e-5)

    def test_certain_distribution_zero_entropy(self):
        probs = np.array([[[1.0, 0.0, 0.0]]])
        ent = shannon_entropy(probs)
        assert ent[0, 0] == pytest.approx(0.0, abs=1e-7)

    def test_nan_propagation(self):
        probs = np.full((2, 2, 3), np.nan)
        ent = shannon_entropy(probs)
        assert np.all(np.isnan(ent))
