"""Benchmark: inference strategies on a large Bayesian network.

Builds a synthetic BN with many evidence roots (large evidence state space),
generates random discretised rasters, and times:

1. the per-combination VariableElimination loop (the only strategy before
   the single-joint-query optimisation), and
2. the single-joint-query conditional-table strategy.

The per-combo loop is timed on a moderate network and extrapolated to the
large one — running it for real would take hours.

Run with:  uv run python examples/benchmark_large_network.py
"""
from __future__ import annotations

import time

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

import geobn.inference as gi


def build_synthetic_bn(n_evidence: int, n_states: int = 3, seed: int = 7):
    """BN with *n_evidence* roots → two mid-layer nodes → one 'risk' node."""
    rng = np.random.default_rng(seed)
    evidence = [f"e{i}" for i in range(n_evidence)]
    half = n_evidence // 2
    edges = [(e, "m0") for e in evidence[:half]]
    edges += [(e, "m1") for e in evidence[half:]]
    edges += [("m0", "risk"), ("m1", "risk")]
    model = DiscreteBayesianNetwork(edges)

    def random_cpd(node, parents):
        card = n_states ** len(parents)
        values = rng.dirichlet(np.ones(n_states), size=card).T
        return TabularCPD(
            node, n_states, values,
            evidence=parents or None,
            evidence_card=[n_states] * len(parents) or None,
            state_names={v: [f"s{k}" for k in range(n_states)] for v in [node, *parents]},
        )

    cpds = [random_cpd(e, []) for e in evidence]
    cpds.append(random_cpd("m0", evidence[:half]))
    cpds.append(random_cpd("m1", evidence[half:]))
    cpds.append(random_cpd("risk", ["m0", "m1"]))
    model.add_cpds(*cpds)
    assert model.check_model()
    return model, evidence


def random_evidence_grids(evidence, n_states, H, W, seed=11):
    rng = np.random.default_rng(seed)
    return {e: rng.integers(0, n_states, (H, W)).astype(np.int16) for e in evidence}


def time_run_inference(model, evidence, n_states, H, W, force_loop):
    grids = random_evidence_grids(evidence, n_states, H, W)
    kwargs = dict(
        model=model,
        evidence_state_grids=grids,
        evidence_state_names={e: [f"s{k}" for k in range(n_states)] for e in evidence},
        query_nodes=["risk"],
        query_state_names={"risk": [f"s{k}" for k in range(n_states)]},
        nodata_mask=np.zeros((H, W), dtype=bool),
    )
    matrix = np.column_stack([grids[e].ravel() for e in evidence])
    n_unique = len(np.unique(matrix, axis=0))

    saved = gi._COMBO_LOOP_THRESHOLD
    gi._COMBO_LOOP_THRESHOLD = 10**12 if force_loop else 0
    try:
        t0 = time.perf_counter()
        gi.run_inference(**kwargs)
        elapsed = time.perf_counter() - t0
    finally:
        gi._COMBO_LOOP_THRESHOLD = saved
    return elapsed, n_unique


def main():
    n_states = 3
    H, W = 500, 500

    print(f"Raster: {H}×{W} = {H * W:,} pixels, {n_states} states per node\n")

    # ── Moderate network: both strategies are feasible ──────────────────
    model, evidence = build_synthetic_bn(n_evidence=6, n_states=n_states)
    t_loop, n_unique = time_run_inference(model, evidence, n_states, H, W, force_loop=True)
    t_joint, _ = time_run_inference(model, evidence, n_states, H, W, force_loop=False)
    per_combo = t_loop / n_unique
    print(f"6 evidence nodes ({n_states**6} possible, {n_unique} observed combos):")
    print(f"  per-combo loop:     {t_loop:8.2f} s  ({per_combo * 1e3:.2f} ms/combo)")
    print(f"  single joint query: {t_joint:8.2f} s  ({t_loop / t_joint:.0f}× faster)\n")

    # ── Large network: loop time extrapolated from the measured rate ────
    model, evidence = build_synthetic_bn(n_evidence=10, n_states=n_states)
    t_joint, n_unique = time_run_inference(model, evidence, n_states, H, W, force_loop=False)
    est_loop = per_combo * n_unique
    print(f"10 evidence nodes ({n_states**10:,} possible, {n_unique:,} observed combos):")
    print(f"  per-combo loop:     {est_loop:8.2f} s  (estimated)")
    print(f"  single joint query: {t_joint:8.2f} s  ({est_loop / t_joint:.0f}× faster)")


if __name__ == "__main__":
    main()
