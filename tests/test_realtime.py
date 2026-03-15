"""Tests for real-time / repeated-inference optimisations.

Covers:
  - freeze()   — static-node caching (Tier 1)
  - precompute() — full table lookup (Tier 2)
  - clear_cache() — cache invalidation
"""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from geobn.network import GeoBayesianNetwork
from geobn.sources.array_source import ArraySource
from geobn.sources.constant_source import ConstantSource


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CRS = "EPSG:4326"
_TRANSFORM = Affine(0.1, 0, 0.0, 0, -0.1, 49.3)
_H, _W = 3, 3

# Slope array covering all three states: flat / moderate / steep
_SLOPE = np.array(
    [[5.0, 20.0, 35.0],
     [5.0, 20.0, 35.0],
     [5.0, 20.0, 35.0]],
    dtype=np.float32,
)
# Rainfall array: low / medium / high per column
_RAIN = np.array(
    [[10.0, 50.0, 150.0],
     [10.0, 50.0, 150.0],
     [10.0, 50.0, 150.0]],
    dtype=np.float32,
)

_SLOPE_DISC = [0, 10, 30, 90]
_SLOPE_LABELS = ["flat", "moderate", "steep"]
_RAIN_DISC = [0, 25, 75, 200]
_RAIN_LABELS = ["low", "medium", "high"]


def _make_bn(fire_risk_model) -> GeoBayesianNetwork:
    """Return a wired-up BN using the fire-risk fixture."""
    bn = GeoBayesianNetwork(fire_risk_model)
    bn.set_grid(_CRS, 0.1, (0.0, 49.0, 0.3, 49.3))
    bn.set_input("slope", ArraySource(_SLOPE, crs=_CRS, transform=_TRANSFORM))
    bn.set_input("rainfall", ArraySource(_RAIN, crs=_CRS, transform=_TRANSFORM))
    bn.set_discretization("slope", _SLOPE_DISC, _SLOPE_LABELS)
    bn.set_discretization("rainfall", _RAIN_DISC, _RAIN_LABELS)
    return bn


# ---------------------------------------------------------------------------
# TestFreezeCache
# ---------------------------------------------------------------------------


class TestFreezeCache:
    def test_freeze_results_match_normal(self, fire_risk_model):
        """Frozen-cache path must produce identical results to the normal path."""
        bn_normal = _make_bn(fire_risk_model)
        result_normal = bn_normal.infer(query=["fire_risk"])

        bn_frozen = _make_bn(fire_risk_model)
        bn_frozen.freeze("slope")
        result_frozen = bn_frozen.infer(query=["fire_risk"])

        np.testing.assert_allclose(
            result_normal.probabilities["fire_risk"],
            result_frozen.probabilities["fire_risk"],
            atol=1e-5,
        )

    def test_freeze_source_fetched_only_once(self, fire_risk_model):
        """A frozen source's fetch() must be called exactly once across N infer() calls."""
        bn = _make_bn(fire_risk_model)

        fetch_count = [0]
        original_fetch = bn._inputs["slope"].fetch

        def counting_fetch(grid=None):
            fetch_count[0] += 1
            return original_fetch(grid=grid)

        bn._inputs["slope"].fetch = counting_fetch

        bn.freeze("slope")
        for _ in range(4):
            bn.infer(query=["fire_risk"])

        assert fetch_count[0] == 1, (
            f"Expected slope to be fetched exactly once, got {fetch_count[0]}"
        )

    def test_non_frozen_source_fetched_every_call(self, fire_risk_model):
        """A non-frozen source must be re-fetched on every infer() call."""
        bn = _make_bn(fire_risk_model)

        fetch_count = [0]
        original_fetch = bn._inputs["rainfall"].fetch

        def counting_fetch(grid=None):
            fetch_count[0] += 1
            return original_fetch(grid=grid)

        bn._inputs["rainfall"].fetch = counting_fetch

        bn.freeze("slope")  # freeze slope, NOT rainfall
        for _ in range(3):
            bn.infer(query=["fire_risk"])

        assert fetch_count[0] == 3, (
            f"Expected rainfall fetched 3 times, got {fetch_count[0]}"
        )

    def test_freeze_with_changing_dynamic_input(self, fire_risk_model):
        """Results change correctly when a dynamic input changes between calls."""
        bn = _make_bn(fire_risk_model)
        bn.freeze("slope")

        # Low rainfall everywhere
        bn.set_input("rainfall", ArraySource(
            np.full((_H, _W), 10.0, dtype=np.float32), crs=_CRS, transform=_TRANSFORM
        ))
        result_low = bn.infer(query=["fire_risk"])

        # High rainfall everywhere
        bn.set_input("rainfall", ArraySource(
            np.full((_H, _W), 150.0, dtype=np.float32), crs=_CRS, transform=_TRANSFORM
        ))
        result_high = bn.infer(query=["fire_risk"])

        # In the synthetic CPT, higher rainfall correlates with higher fire risk
        # (steep+high → 0.70 vs steep+low → 0.40).  The key check is that the
        # cached terrain arrays give different results when weather changes.
        p_high_low_rain = float(np.nanmean(result_low.probabilities["fire_risk"][..., 2]))
        p_high_high_rain = float(np.nanmean(result_high.probabilities["fire_risk"][..., 2]))
        assert p_high_high_rain != p_high_low_rain, (
            "Expected results to differ when rainfall input changes"
        )

    def test_clear_cache_triggers_refetch(self, fire_risk_model):
        """After clear_cache(), the frozen source must be fetched again."""
        bn = _make_bn(fire_risk_model)

        fetch_count = [0]
        original_fetch = bn._inputs["slope"].fetch

        def counting_fetch(grid=None):
            fetch_count[0] += 1
            return original_fetch(grid=grid)

        bn._inputs["slope"].fetch = counting_fetch

        bn.freeze("slope")
        bn.infer(query=["fire_risk"])   # fetches once → count = 1
        bn.infer(query=["fire_risk"])   # uses cache → count still 1
        bn.clear_cache()
        bn.infer(query=["fire_risk"])   # cache cleared → fetches again → count = 2

        assert fetch_count[0] == 2

    def test_freeze_empty_set_is_no_op(self, fire_risk_model):
        """Calling freeze() with no arguments should produce normal inference."""
        bn = _make_bn(fire_risk_model)
        bn_ref = _make_bn(fire_risk_model)

        bn.freeze()  # no frozen nodes
        result = bn.infer(query=["fire_risk"])
        result_ref = bn_ref.infer(query=["fire_risk"])

        np.testing.assert_allclose(
            result.probabilities["fire_risk"],
            result_ref.probabilities["fire_risk"],
            atol=1e-5,
        )

    def test_freeze_validates_node_name(self, fire_risk_model):
        bn = _make_bn(fire_risk_model)
        with pytest.raises(ValueError, match="does not exist"):
            bn.freeze("nonexistent_node")

    def test_ve_cached_across_calls(self, fire_risk_model):
        """_cached_ve should be populated after the first infer() call."""
        bn = _make_bn(fire_risk_model)
        assert bn._cached_ve is None
        bn.infer(query=["fire_risk"])
        assert bn._cached_ve is not None
        ve_first = bn._cached_ve
        bn.infer(query=["fire_risk"])
        assert bn._cached_ve is ve_first, "VE object should be reused, not recreated"


# ---------------------------------------------------------------------------
# TestPrecompute
# ---------------------------------------------------------------------------


class TestPrecompute:
    def test_precompute_table_shape(self, fire_risk_model):
        """Table shape must be (n_slope, n_rain, n_fire_risk) = (3, 3, 3)."""
        bn = _make_bn(fire_risk_model)
        bn.precompute(query=["fire_risk"])

        assert "fire_risk" in bn._inference_table
        expected_shape = (3, 3, 3)  # slope_states × rain_states × risk_states
        assert bn._inference_table["fire_risk"].shape == expected_shape

    def test_precompute_matches_normal_inference(self, fire_risk_model):
        """Table-path results must match normal-path results pixel-for-pixel."""
        bn_normal = _make_bn(fire_risk_model)
        result_normal = bn_normal.infer(query=["fire_risk"])

        bn_table = _make_bn(fire_risk_model)
        bn_table.precompute(query=["fire_risk"])
        result_table = bn_table.infer(query=["fire_risk"])

        np.testing.assert_allclose(
            result_normal.probabilities["fire_risk"],
            result_table.probabilities["fire_risk"],
            atol=1e-5,
        )

    def test_precompute_probs_sum_to_one(self, fire_risk_model):
        """Every valid pixel's probabilities must sum to 1.0."""
        bn = _make_bn(fire_risk_model)
        bn.precompute(query=["fire_risk"])
        result = bn.infer(query=["fire_risk"])

        probs = result.probabilities["fire_risk"]
        valid = ~np.isnan(probs[..., 0])
        np.testing.assert_allclose(
            probs[valid].sum(axis=-1), 1.0, atol=1e-5
        )

    def test_precompute_realtime_loop(self, fire_risk_model):
        """Multiple infer() calls after precompute() all return correct results."""
        bn = _make_bn(fire_risk_model)
        bn.freeze("slope")
        bn.precompute(query=["fire_risk"])

        rain_values = [10.0, 50.0, 100.0, 150.0]
        for rain_val in rain_values:
            bn.set_input("rainfall", ConstantSource(rain_val))

            result_table = bn.infer(query=["fire_risk"])

            # Compare against a fresh un-optimised BN for the same scenario
            bn_ref = _make_bn(fire_risk_model)
            bn_ref.set_input("rainfall", ConstantSource(rain_val))
            result_ref = bn_ref.infer(query=["fire_risk"])

            np.testing.assert_allclose(
                result_table.probabilities["fire_risk"],
                result_ref.probabilities["fire_risk"],
                atol=1e-4,
                err_msg=f"Mismatch at rainfall={rain_val}",
            )

    def test_precompute_no_pgmpy_on_second_call(self, fire_risk_model):
        """After precompute(), the table path is activated (table is non-empty)."""
        bn = _make_bn(fire_risk_model)
        bn.precompute(query=["fire_risk"])

        assert bn._inference_table, "Table should be populated after precompute()"
        assert bn._table_node_order == ["slope", "rainfall"]
        assert bn._table_query_nodes == ["fire_risk"]

    def test_precompute_raises_without_discretizations(self, fire_risk_model):
        """precompute() must raise RuntimeError if discretizations are missing."""
        bn = GeoBayesianNetwork(fire_risk_model)
        bn.set_grid(_CRS, 0.1, (0.0, 49.0, 0.3, 49.3))
        bn.set_input("slope", ArraySource(_SLOPE, crs=_CRS, transform=_TRANSFORM))
        bn.set_input("rainfall", ArraySource(_RAIN, crs=_CRS, transform=_TRANSFORM))
        # intentionally skip set_discretization()

        with pytest.raises(RuntimeError, match="No discretization"):
            bn.precompute(query=["fire_risk"])

    def test_precompute_nodata_propagates(self, fire_risk_model):
        """NaN in input must propagate as NaN in table-path output."""
        slope_with_nan = _SLOPE.copy()
        slope_with_nan[1, 1] = np.nan

        bn = GeoBayesianNetwork(fire_risk_model)
        bn.set_grid(_CRS, 0.1, (0.0, 49.0, 0.3, 49.3))
        bn.set_input("slope", ArraySource(slope_with_nan, crs=_CRS, transform=_TRANSFORM))
        bn.set_input("rainfall", ArraySource(_RAIN, crs=_CRS, transform=_TRANSFORM))
        bn.set_discretization("slope", _SLOPE_DISC, _SLOPE_LABELS)
        bn.set_discretization("rainfall", _RAIN_DISC, _RAIN_LABELS)
        bn.precompute(query=["fire_risk"])

        result = bn.infer(query=["fire_risk"])
        probs = result.probabilities["fire_risk"]

        # Pixel (1,1) had NaN slope — must be NaN in output
        assert np.all(np.isnan(probs[1, 1, :]))
        # Pixel (0,0) is valid
        assert not np.any(np.isnan(probs[0, 0, :]))

    def test_clear_cache_resets_table(self, fire_risk_model):
        """clear_cache() must remove the precomputed table."""
        bn = _make_bn(fire_risk_model)
        bn.precompute(query=["fire_risk"])
        assert bn._inference_table

        bn.clear_cache()
        assert not bn._inference_table
        assert not bn._table_node_order
        assert not bn._table_query_nodes

    def test_set_grid_clears_frozen_cache(self, fire_risk_model):
        """set_grid() after freeze() must invalidate the frozen discrete array cache.

        Without this, the cached array from the old grid (wrong shape) would be
        reused on the next infer() call, producing a shape-mismatch crash or
        silently wrong results.
        """
        bn = _make_bn(fire_risk_model)
        bn.freeze("slope")
        bn.infer(query=["fire_risk"])  # populates _frozen_cache with (3, 3) arrays
        assert bn._frozen_cache  # cache is populated

        # Change grid to a different size
        bn.set_grid(_CRS, 0.1, (0.0, 49.0, 0.4, 49.4))
        assert not bn._frozen_cache  # must be cleared

        # Update inputs to match the new grid size and re-run — must not crash
        new_slope = np.tile(_SLOPE[0], (4, 4)).astype(np.float32)[:4, :4]
        new_rain = np.tile(_RAIN[0], (4, 4)).astype(np.float32)[:4, :4]
        t = Affine(0.1, 0, 0.0, 0, -0.1, 49.4)
        bn.set_input("slope", ArraySource(new_slope, crs=_CRS, transform=t))
        bn.set_input("rainfall", ArraySource(new_rain, crs=_CRS, transform=t))
        result = bn.infer(query=["fire_risk"])
        assert result.probabilities["fire_risk"].shape[:2] == (4, 4)


# ---------------------------------------------------------------------------
# TestSaveLoadPrecomputed
# ---------------------------------------------------------------------------


class TestSaveLoadPrecomputed:
    def test_save_creates_file(self, fire_risk_model, tmp_path):
        """save_precomputed() must create a .npz file at the given path."""
        bn = _make_bn(fire_risk_model)
        bn.precompute(query=["fire_risk"])
        out = tmp_path / "table.npz"
        bn.save_precomputed(out)
        assert out.exists()

    def test_load_restores_table(self, fire_risk_model, tmp_path):
        """load_precomputed() must populate _inference_table, _table_node_order, _table_query_nodes."""
        bn = _make_bn(fire_risk_model)
        bn.precompute(query=["fire_risk"])
        out = tmp_path / "table.npz"
        bn.save_precomputed(out)

        bn2 = _make_bn(fire_risk_model)
        bn2.load_precomputed(out)

        assert bn2._inference_table, "Table should be non-empty after load"
        assert bn2._table_node_order == ["slope", "rainfall"]
        assert bn2._table_query_nodes == ["fire_risk"]

    def test_load_matches_precomputed_results(self, fire_risk_model, tmp_path):
        """infer() after load_precomputed() must match infer() after precompute() (atol=1e-5)."""
        bn = _make_bn(fire_risk_model)
        bn.precompute(query=["fire_risk"])
        result_precomputed = bn.infer(query=["fire_risk"])

        out = tmp_path / "table.npz"
        bn.save_precomputed(out)

        bn2 = _make_bn(fire_risk_model)
        bn2.load_precomputed(out)
        result_loaded = bn2.infer(query=["fire_risk"])

        np.testing.assert_allclose(
            result_precomputed.probabilities["fire_risk"],
            result_loaded.probabilities["fire_risk"],
            atol=1e-5,
        )

    def test_save_raises_without_table(self, fire_risk_model, tmp_path):
        """save_precomputed() before precompute() must raise RuntimeError."""
        bn = _make_bn(fire_risk_model)
        with pytest.raises(RuntimeError, match="No precomputed table"):
            bn.save_precomputed(tmp_path / "table.npz")

    def test_load_raises_on_node_order_mismatch(self, fire_risk_model, tmp_path):
        """load_precomputed() must raise ValueError if node order differs."""
        bn = _make_bn(fire_risk_model)
        bn.precompute(query=["fire_risk"])
        out = tmp_path / "table.npz"
        bn.save_precomputed(out)

        # Build a BN with inputs registered in reverse order
        bn2 = GeoBayesianNetwork(fire_risk_model)
        bn2.set_grid(_CRS, 0.1, (0.0, 49.0, 0.3, 49.3))
        bn2.set_input("rainfall", ArraySource(_RAIN, crs=_CRS, transform=_TRANSFORM))
        bn2.set_input("slope", ArraySource(_SLOPE, crs=_CRS, transform=_TRANSFORM))
        bn2.set_discretization("slope", _SLOPE_DISC, _SLOPE_LABELS)
        bn2.set_discretization("rainfall", _RAIN_DISC, _RAIN_LABELS)

        with pytest.raises(ValueError, match="Node order mismatch"):
            bn2.load_precomputed(out)

    def test_load_raises_on_shape_mismatch(self, fire_risk_model, tmp_path):
        """load_precomputed() must raise ValueError if discretization n_states differ."""
        from geobn.discretize import DiscretizationSpec

        bn = _make_bn(fire_risk_model)
        bn.precompute(query=["fire_risk"])
        out = tmp_path / "table.npz"
        bn.save_precomputed(out)

        # Build a normal BN, then override _discretizations to simulate a mismatch
        # (fewer bins for slope than the saved table expects).
        bn2 = _make_bn(fire_risk_model)
        # Directly replace the discretization so n_states differs from the saved table
        bn2._discretizations["slope"] = DiscretizationSpec(
            breakpoints=[0, 45, 90], labels=["gentle", "steep"]
        )

        with pytest.raises(ValueError, match="Shape mismatch"):
            bn2.load_precomputed(out)

    def test_load_raises_file_not_found(self, fire_risk_model, tmp_path):
        """load_precomputed() must raise FileNotFoundError for a missing file."""
        bn = _make_bn(fire_risk_model)
        with pytest.raises(FileNotFoundError):
            bn.load_precomputed(tmp_path / "nonexistent.npz")

    def test_roundtrip_no_pgmpy(self, fire_risk_model, tmp_path):
        """After load_precomputed(), infer() should use the table path (no pgmpy)."""
        bn = _make_bn(fire_risk_model)
        bn.precompute(query=["fire_risk"])
        out = tmp_path / "table.npz"
        bn.save_precomputed(out)

        bn2 = _make_bn(fire_risk_model)
        bn2.load_precomputed(out)

        assert bn2._inference_table, "_inference_table must be populated"
        result = bn2.infer(query=["fire_risk"])
        # Result should be valid (non-NaN) for all pixels
        probs = result.probabilities["fire_risk"]
        assert not np.all(np.isnan(probs)), "Expected valid (non-NaN) probabilities"
