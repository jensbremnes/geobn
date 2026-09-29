"""Tests for NoData inputs that only blank the query nodes they can affect.

A missing input makes a query node NaN when it is d-connected to that node
given the inputs observed at the pixel.  Other query nodes get the posterior
given the observed inputs, the same on every inference path.
"""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from test_query_strategy import _random_model

import geobn.inference as inference_module
from geobn import ArraySource, GeoBayesianNetwork
from geobn.inference import (
    _query_marginals,
    _root_priors,
    blank_masks,
    build_conditional_table,
    make_blanked_by,
    run_inference,
    run_inference_from_table,
)

_TRANSFORM = Affine(1.0, 0, 0.0, 0, -1.0, 2.0)
_CRS = "EPSG:32632"


def _binary_cpd(node, parents, rng):
    values = rng.uniform(0.1, 0.9, (1, 2 ** len(parents)))
    values = np.vstack([values, 1 - values])
    return TabularCPD(
        node, 2, values,
        evidence=parents or None,
        evidence_card=[2] * len(parents) or None,
        state_names={n: [f"{n.lower()}0", f"{n.lower()}1"] for n in [node, *parents]},
    )


@pytest.fixture
def relevance_model() -> DiscreteBayesianNetwork:
    """Inputs Z, X, M, X2, C; query nodes Q1, Q2.

    - Z is not connected to anything.
    - Chain X -> M -> Q1: with M observed, X cannot affect Q1.
    - Collider X2 -> C <- Y2 -> Q2: with C observed, X2 affects Q2.  With C
      unobserved the collider blocks X2, but C itself then reaches Q2.
    """
    rng = np.random.default_rng(3)
    edges = [("X", "M"), ("M", "Q1"), ("X2", "C"), ("Y2", "C"), ("Y2", "Q2")]
    model = DiscreteBayesianNetwork(edges)
    model.add_node("Z")
    for node in model.nodes():
        model.add_cpds(_binary_cpd(node, sorted(model.predecessors(node)), rng))
    model.check_model()
    return model


def _names(model, nodes):
    return {n: list(model.get_cpds(n).state_names[n]) for n in nodes}


def _expected(model, grids, query):
    """Pixel by pixel: NaN if a missing input is d-connected to q, else P(q | observed)."""
    ve = VariableElimination(model)
    priors = _root_priors(model)
    nodes = list(grids)
    names = _names(model, nodes)
    shape = np.shape(grids[nodes[0]])
    out = {q: np.full((*shape, model.get_cardinality(q)), np.nan) for q in query}
    for idx in np.ndindex(shape):
        observed = {n: names[n][grids[n][idx]] for n in nodes if grids[n][idx] >= 0}
        missing = [n for n in nodes if grids[n][idx] < 0]
        free = [
            q for q in query
            if not any(model.is_dconnected(x, q, observed=list(observed)) for x in missing)
        ]
        if free:
            marginals = _query_marginals(ve, model, free, observed, priors)
            for q in free:
                out[q][idx] = marginals[q]
    return out


def _all_paths(model, grids, query, monkeypatch):
    """Results of the per-combo loop, the joint table and the precomputed-table lookup."""
    nodes = list(grids)
    kwargs = dict(
        model=model,
        evidence_state_grids=grids,
        evidence_state_names=_names(model, nodes),
        query_nodes=query,
        query_state_names=_names(model, query),
    )
    with monkeypatch.context() as m:
        m.setattr(inference_module, "_COMBO_LOOP_THRESHOLD", 10**9)
        loop = run_inference(**kwargs)
    with monkeypatch.context() as m:
        m.setattr(inference_module, "_COMBO_LOOP_THRESHOLD", 0)
        joint = run_inference(**kwargs)
    table = run_inference_from_table(
        build_conditional_table(model, nodes, query),
        nodes,
        grids,
        blanked_by=make_blanked_by(model, nodes, query),
    )
    return {"loop": loop, "joint": joint, "table": table}


def _assert_matches(results, expected):
    for path, result in results.items():
        for q, exp in expected.items():
            np.testing.assert_allclose(
                result[q], exp, atol=1e-5, equal_nan=True, err_msg=f"{path} path, {q}"
            )


# Columns are pixels; -1 is NoData.
_CASES = {
    #           Z   X   M  X2   C
    "none":   [ 0,  1,  0,  1,  0],
    "Z":      [-1,  1,  0,  1,  0],
    "X":      [ 0, -1,  1,  0,  1],
    "M":      [ 1,  0, -1,  1,  1],
    "X2":     [ 1,  1,  1, -1,  0],
    "X2+C":   [ 0,  0,  1, -1, -1],
    "all":    [-1, -1, -1, -1, -1],
}
_INPUTS = ["Z", "X", "M", "X2", "C"]


def _case_grids():
    matrix = np.array(list(_CASES.values()), dtype=np.int16)   # (pixels, inputs)
    return {n: matrix[:, i][np.newaxis, :] for i, n in enumerate(_INPUTS)}


class TestRelevance:
    def test_blanked_by(self, relevance_model):
        blanked_by = make_blanked_by(relevance_model, _INPUTS, ["Q1", "Q2"])
        assert blanked_by(frozenset()) == frozenset()
        assert blanked_by(frozenset({"Z"})) == frozenset()
        assert blanked_by(frozenset({"X"})) == frozenset()
        assert blanked_by(frozenset({"M"})) == {"Q1"}
        assert blanked_by(frozenset({"X2"})) == {"Q2"}
        assert blanked_by(frozenset({"X2", "C"})) == {"Q2"}      # C itself reaches Q2
        assert blanked_by(frozenset({"X", "M", "X2", "C", "Z"})) == {"Q1", "Q2"}

        # C is not an input: the unobserved collider blocks X2 from Q2.
        only_x2 = make_blanked_by(relevance_model, ["X2"], ["Q2"])
        assert only_x2(frozenset({"X2"})) == frozenset()

    def test_blank_masks(self, relevance_model):
        grids = _case_grids()
        blanked_by = make_blanked_by(relevance_model, _INPUTS, ["Q1", "Q2"])
        masks = blank_masks(grids, _INPUTS, ["Q1", "Q2"], blanked_by)
        cases = list(_CASES)
        assert [c for c, b in zip(cases, masks["Q1"][0]) if b] == ["M", "all"]
        assert [c for c, b in zip(cases, masks["Q2"][0]) if b] == ["X2", "X2+C", "all"]

        without = blank_masks(grids, _INPUTS, ["Q1"])
        assert [c for c, b in zip(cases, without["Q1"][0]) if b] == cases[1:]

    def test_cases_on_every_path(self, relevance_model, monkeypatch):
        grids = _case_grids()
        results = _all_paths(relevance_model, grids, ["Q1", "Q2"], monkeypatch)
        _assert_matches(results, _expected(relevance_model, grids, ["Q1", "Q2"]))

        column = {c: i for i, c in enumerate(_CASES)}
        for result in results.values():
            q1, q2 = result["Q1"][0], result["Q2"][0]
            for case in ["none", "Z", "X", "X2", "X2+C"]:
                assert np.isfinite(q1[column[case]]).all(), case
            for case in ["M", "all"]:
                assert np.isnan(q1[column[case]]).all(), case
            for case in ["none", "Z", "X", "M"]:
                assert np.isfinite(q2[column[case]]).all(), case
            for case in ["X2", "X2+C", "all"]:
                assert np.isnan(q2[column[case]]).all(), case

    def test_irrelevant_input_does_not_change_the_posterior(self, relevance_model, monkeypatch):
        """Q1 given M is the same whether X is observed or missing."""
        grids = {n: np.zeros((1, 3), dtype=np.int16) for n in _INPUTS}
        grids["X"][0] = [0, 1, -1]
        for result in _all_paths(relevance_model, grids, ["Q1"], monkeypatch).values():
            q1 = result["Q1"][0]
            np.testing.assert_allclose(q1[0], q1[2], atol=1e-6)
            np.testing.assert_allclose(q1[1], q1[2], atol=1e-6)

    def test_table_without_blanked_by_blanks_everything(self, relevance_model):
        grids = _case_grids()
        table = build_conditional_table(relevance_model, _INPUTS, ["Q1"])
        result = run_inference_from_table(table, _INPUTS, grids)
        assert np.isfinite(result["Q1"][0, 0]).all()
        assert np.isnan(result["Q1"][0, 1:]).all()

    @pytest.mark.parametrize("seed", range(30))
    def test_random_models(self, seed, monkeypatch):
        rng = np.random.default_rng(500 + seed)
        model = _random_model(rng, int(rng.integers(3, 8)), with_zeros=seed % 3 == 0)
        nodes = list(rng.permutation(list(model.nodes())))
        n_inputs = int(rng.integers(1, len(nodes)))
        inputs, query = nodes[:n_inputs], nodes[n_inputs:]

        grids = {}
        for n in inputs:
            g = rng.integers(0, model.get_cardinality(n), (4, 5)).astype(np.int16)
            g[rng.random(g.shape) < 0.3] = -1
            grids[n] = g

        results = _all_paths(model, grids, query, monkeypatch)
        _assert_matches(results, _expected(model, grids, query))


@pytest.fixture
def impossible_model() -> DiscreteBayesianNetwork:
    """A (prior [1, 0]) and B -> C;  D -> E, independent of A."""
    model = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("D", "E")])
    model.add_cpds(
        TabularCPD("A", 2, [[1.0], [0.0]], state_names={"A": ["a0", "a1"]}),
        TabularCPD("B", 2, [[0.4], [0.6]], state_names={"B": ["b0", "b1"]}),
        TabularCPD("D", 2, [[0.5], [0.5]], state_names={"D": ["d0", "d1"]}),
        TabularCPD(
            "C", 2, [[0.9, 0.5, 0.3, 0.1], [0.1, 0.5, 0.7, 0.9]],
            evidence=["A", "B"], evidence_card=[2, 2],
            state_names={"C": ["c0", "c1"], "A": ["a0", "a1"], "B": ["b0", "b1"]},
        ),
        TabularCPD(
            "E", 2, [[0.8, 0.2], [0.2, 0.8]], evidence=["D"], evidence_card=[2],
            state_names={"E": ["e0", "e1"], "D": ["d0", "d1"]},
        ),
    )
    model.check_model()
    return model


def _bn(model, grids: dict[str, list[list[float]]]) -> GeoBayesianNetwork:
    bn = GeoBayesianNetwork(model)
    for node, values in grids.items():
        arr = np.asarray(values, dtype=float)
        bn.set_input(node, ArraySource(arr, crs=_CRS, transform=_TRANSFORM))
        bn.set_discretization(node, [-0.5, 0.5, 1.5])
    return bn


class TestPublicAPI:
    @pytest.mark.parametrize("use_table", [False, True])
    def test_each_query_node_has_its_own_holes(self, use_table):
        """S1 -> Q1 and S2 -> Q2: a hole in S1 leaves Q2 intact, and the reverse."""
        rng = np.random.default_rng(0)
        model = DiscreteBayesianNetwork([("S1", "Q1"), ("S2", "Q2")])
        for node in model.nodes():
            model.add_cpds(_binary_cpd(node, sorted(model.predecessors(node)), rng))
        bn = _bn(model, {"S1": [[np.nan, 1], [0, 1]], "S2": [[0, 1], [1, np.nan]]})
        if use_table:
            bn.precompute(query=["Q1", "Q2"])
        result = bn.infer(query=["Q1", "Q2"])

        q1_nan = np.isnan(result.probabilities["Q1"]).all(axis=-1)
        q2_nan = np.isnan(result.probabilities["Q2"]).all(axis=-1)
        np.testing.assert_array_equal(q1_nan, [[True, False], [False, False]])
        np.testing.assert_array_equal(q2_nan, [[False, False], [False, True]])
        assert not np.isnan(result.probabilities["Q1"][~q1_nan]).any()

    @pytest.mark.parametrize("use_table", [False, True])
    def test_impossible_observed_evidence_warns(self, impossible_model, use_table):
        """Blanked query nodes are not reported; P(observed) = 0 is."""
        grids = {
            "A": [[1, 1, 0]],
            "B": [[np.nan, 0, 0]],   # pixel 0: B missing -> C blanked, E impossible
            "D": [[1, np.nan, 1]],   # pixel 1: D missing -> E blanked, C impossible
        }
        bn = _bn(impossible_model, grids)
        if use_table:
            bn.precompute(query=["C", "E"])
        with pytest.warns(UserWarning, match=r"2 pixel\(s\).*A='a1'.*prior probability of 0"):
            result = bn.infer(query=["C", "E"])
        for q in ["C", "E"]:
            assert np.isnan(result.probabilities[q][0, :2]).all()
            assert np.isfinite(result.probabilities[q][0, 2]).all()

    def test_no_warning_for_blanked_nodes_only(self, impossible_model, recwarn):
        bn = _bn(impossible_model, {"A": [[0, np.nan]], "B": [[0, 0]], "D": [[np.nan, 1]]})
        result = bn.infer(query=["C", "E"])
        assert not [w for w in recwarn if "probability zero" in str(w.message)]
        assert np.isnan(result.probabilities["E"][0, 0]).all()
        assert np.isfinite(result.probabilities["C"][0, 0]).all()
        assert np.isnan(result.probabilities["C"][0, 1]).all()
        assert np.isfinite(result.probabilities["E"][0, 1]).all()

    def test_query_batch_and_point(self, relevance_model):
        bn = _bn(relevance_model, {n: [[0, 1], [1, 0]] for n in _INPUTS})
        bn.precompute(query=["Q1", "Q2"])

        grids = _case_grids()
        evidence = {n: np.where(grids[n][0] < 0, np.nan, grids[n][0]).astype(float) for n in _INPUTS}
        out = bn.query_batch(evidence)
        expected = _expected(relevance_model, {n: g[0] for n, g in grids.items()}, ["Q1", "Q2"])
        for q in ["Q1", "Q2"]:
            np.testing.assert_allclose(out[q], expected[q], atol=1e-5, equal_nan=True)

        point = bn.query_point({"Z": 0, "X": np.nan, "M": 1, "X2": 0, "C": 1})
        q1 = bn.query_point({"Z": 0, "X": 0, "M": 1, "X2": 0, "C": 1})["Q1"]
        assert point["Q1"] == pytest.approx(q1, abs=1e-6)
        assert not np.isnan(list(point["Q2"].values())).any()
