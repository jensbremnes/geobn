"""Tests for the inference engine."""
from __future__ import annotations

import numpy as np
import pytest

from geobn.inference import run_inference, shannon_entropy


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
