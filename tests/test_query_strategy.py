"""Tests for the per-combination query strategy and impossible evidence.

Within one evidence combination, query nodes are requested from pgmpy in a
single call when their joint table is small, and one by one otherwise.  Both
must give the same posteriors, and evidence with probability zero must give
NaN for every query node on every inference path.
"""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

import geobn.inference as inference_module
import geobn.network as network_module
from geobn import ArraySource, GeoBayesianNetwork
from geobn.inference import _query_marginals, _root_priors, run_inference


def _states(node: str, n: int) -> list[str]:
    return [f"{node}{i}" for i in range(n)]


def _random_model(rng: np.random.Generator, n_nodes: int, with_zeros: bool) -> DiscreteBayesianNetwork:
    """Random DAG over X0..Xn with 2-3 states per node and random CPDs."""
    names = [f"X{i}" for i in range(n_nodes)]
    cards = {v: int(rng.integers(2, 4)) for v in names}
    edges = [(names[i], names[j]) for j in range(n_nodes) for i in range(j) if rng.random() < 0.35]
    model = DiscreteBayesianNetwork(edges)
    model.add_nodes_from(names)
    for v in names:
        parents = [p for p, c in edges if c == v]
        n_cols = int(np.prod([cards[p] for p in parents])) if parents else 1
        values = rng.random((cards[v], n_cols))
        if with_zeros:
            values[rng.random(values.shape) < 0.3] = 0.0
            values[:, values.sum(axis=0) == 0] = 1.0
        values /= values.sum(axis=0)
        model.add_cpds(TabularCPD(
            v, cards[v], values,
            evidence=parents or None,
            evidence_card=[cards[p] for p in parents] or None,
            state_names={x: _states(x, cards[x]) for x in [v, *parents]},
        ))
    model.check_model()
    return model


@pytest.fixture
def impossible_root_model() -> DiscreteBayesianNetwork:
    """A (prior [1, 0]) and B -> C;  D -> E, independent of A.

    Observing A = a1 is impossible.  E is d-separated from A, so pgmpy on its
    own returns E's posterior as if the evidence were fine.
    """
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


@pytest.fixture
def impossible_non_root_model(impossible_root_model) -> DiscreteBayesianNetwork:
    """As ``impossible_root_model``, but P(C = c1 | a0, b0) = 0.

    Observing the non-root node C together with A = a0, B = b0 is then
    impossible, which a product of root priors cannot detect.
    """
    model = impossible_root_model.copy()
    model.remove_cpds(model.get_cpds("C"))
    model.add_cpds(TabularCPD(
        "C", 2, [[1.0, 0.5, 0.3, 0.1], [0.0, 0.5, 0.7, 0.9]],
        evidence=["A", "B"], evidence_card=[2, 2],
        state_names={"C": ["c0", "c1"], "A": ["a0", "a1"], "B": ["b0", "b1"]},
    ))
    model.check_model()
    return model


def _impossible_grid_kwargs(model, query_nodes):
    """Two pixels: pixel (0, 0) observes A = a1 (impossible), pixel (0, 1) A = a0."""
    return dict(
        model=model,
        evidence_state_grids={
            "A": np.array([[1, 0]], dtype=np.int16),
            "B": np.array([[0, 1]], dtype=np.int16),
            "D": np.array([[1, 1]], dtype=np.int16),
        },
        evidence_state_names={"A": ["a0", "a1"], "B": ["b0", "b1"], "D": ["d0", "d1"]},
        query_nodes=query_nodes,
        query_state_names={q: list(model.get_cpds(q).state_names[q]) for q in query_nodes},
        nodata_mask=np.zeros((1, 2), dtype=bool),
    )


class TestQueryStrategyEquivalence:
    @pytest.mark.parametrize("seed", range(50))
    def test_combined_and_per_node_agree(self, seed, monkeypatch):
        rng = np.random.default_rng(seed)
        model = _random_model(rng, int(rng.integers(3, 9)), with_zeros=seed % 2 == 1)
        ve = VariableElimination(model)
        nodes = list(rng.permutation(list(model.nodes())))
        n_evidence = int(rng.integers(0, len(nodes)))
        evidence = {
            n: model.get_cpds(n).state_names[n][rng.integers(model.get_cardinality(n))]
            for n in nodes[:n_evidence]
        }
        query = nodes[n_evidence:]
        priors = _root_priors(model)

        monkeypatch.setattr(inference_module, "_MAX_JOINT_QUERY_CELLS", 10**12)
        combined = _query_marginals(ve, model, query, evidence, priors)
        monkeypatch.setattr(inference_module, "_MAX_JOINT_QUERY_CELLS", 0)
        per_node = _query_marginals(ve, model, query, evidence, priors)

        for q in query:
            np.testing.assert_allclose(combined[q], per_node[q], atol=1e-6, equal_nan=True)
            if not np.isnan(combined[q]).any():
                np.testing.assert_allclose(combined[q].sum(), 1.0, atol=1e-5)

    @pytest.mark.parametrize("seed", range(20))
    def test_possible_evidence_matches_plain_pgmpy(self, seed):
        """When P(evidence) > 0, results equal a direct pgmpy query."""
        rng = np.random.default_rng(1000 + seed)
        model = _random_model(rng, int(rng.integers(3, 8)), with_zeros=False)
        ve = VariableElimination(model)
        roots = [n for n in model.nodes() if not list(model.predecessors(n))]
        evidence = {n: model.get_cpds(n).state_names[n][0] for n in roots[:2]}
        query = [n for n in model.nodes() if n not in evidence]
        result = _query_marginals(ve, model, query, evidence, _root_priors(model))
        for q in query:
            expected = ve.query([q], evidence=evidence, show_progress=False).values
            np.testing.assert_allclose(result[q], expected, atol=1e-6)


class TestQueryStrategySelection:
    def _count_queries(self, model, monkeypatch, threshold):
        monkeypatch.setattr(inference_module, "_MAX_JOINT_QUERY_CELLS", threshold)
        ve = VariableElimination(model)
        calls = []
        original = ve.query

        def spy(variables, *args, **kwargs):
            calls.append(list(variables))
            return original(variables, *args, **kwargs)

        monkeypatch.setattr(ve, "query", spy)
        _query_marginals(ve, model, ["fire_risk", "rainfall"], {"slope": "flat"}, _root_priors(model))
        return calls

    def test_small_joint_uses_one_call(self, fire_risk_model, monkeypatch):
        calls = self._count_queries(fire_risk_model, monkeypatch, threshold=9)
        assert calls == [["fire_risk", "rainfall"]]

    def test_large_joint_queries_each_node(self, fire_risk_model, monkeypatch):
        calls = self._count_queries(fire_risk_model, monkeypatch, threshold=8)
        assert calls == [["fire_risk"], ["rainfall"]]

    def test_duplicate_query_nodes(self, fire_risk_model):
        kwargs = dict(
            model=fire_risk_model,
            evidence_state_grids={"slope": np.array([[0, 2]], dtype=np.int16)},
            evidence_state_names={"slope": ["flat", "moderate", "steep"]},
            query_state_names={"fire_risk": ["low", "medium", "high"]},
            nodata_mask=np.zeros((1, 2), dtype=bool),
        )
        once = run_inference(query_nodes=["fire_risk"], **kwargs)
        twice = run_inference(query_nodes=["fire_risk", "fire_risk"], **kwargs)
        np.testing.assert_allclose(twice["fire_risk"], once["fire_risk"])


class TestImpossibleEvidence:
    @pytest.mark.parametrize("query_nodes", [["E"], ["C"], ["C", "E"]])
    @pytest.mark.parametrize("max_joint_cells", [0, 10**12])
    def test_per_combo_loop(self, impossible_root_model, monkeypatch, query_nodes, max_joint_cells):
        monkeypatch.setattr(inference_module, "_MAX_JOINT_QUERY_CELLS", max_joint_cells)
        result = run_inference(**_impossible_grid_kwargs(impossible_root_model, query_nodes))
        for q in query_nodes:
            assert np.isnan(result[q][0, 0]).all(), f"{q} should be NaN for impossible evidence"
            assert not np.isnan(result[q][0, 1]).any()

    def test_joint_table_path(self, impossible_root_model, monkeypatch):
        monkeypatch.setattr(inference_module, "_COMBO_LOOP_THRESHOLD", 0)
        result = run_inference(**_impossible_grid_kwargs(impossible_root_model, ["C", "E"]))
        for q in ["C", "E"]:
            assert np.isnan(result[q][0, 0]).all()
            assert not np.isnan(result[q][0, 1]).any()

    def test_paths_agree_on_possible_pixels(self, impossible_root_model, monkeypatch):
        loop = run_inference(**_impossible_grid_kwargs(impossible_root_model, ["C", "E"]))
        monkeypatch.setattr(inference_module, "_COMBO_LOOP_THRESHOLD", 0)
        table = run_inference(**_impossible_grid_kwargs(impossible_root_model, ["C", "E"]))
        for q in ["C", "E"]:
            np.testing.assert_allclose(loop[q], table[q], atol=1e-6, equal_nan=True)

    @pytest.mark.parametrize("force_fallback", [False, True])
    def test_precompute(self, impossible_root_model, monkeypatch, force_fallback):
        if force_fallback:
            monkeypatch.setattr(network_module, "build_conditional_table", lambda *a, **k: None)
        bn = GeoBayesianNetwork(impossible_root_model)
        transform = Affine(1.0, 0, 0.0, 0, -1.0, 2.0)
        for node in ["A", "B", "D"]:
            bn.set_input(node, ArraySource(np.zeros((2, 2)), crs="EPSG:32632", transform=transform))
        bn.precompute(query=["C", "E"])

        impossible = bn.query_batch({"A": "a1", "B": "b0", "D": "d1"})
        possible = bn.query_batch({"A": "a0", "B": "b0", "D": "d1"})
        for q in ["C", "E"]:
            assert np.isnan(impossible[q]).all()
            assert not np.isnan(possible[q]).any()

    def test_non_root_evidence_chain_rule(self, impossible_non_root_model):
        """Evidence C = c1 with A = a0, B = b0 is possible; with a zero CPT entry it is not."""
        model = impossible_non_root_model
        ve = VariableElimination(model)
        priors = _root_priors(model)

        impossible = _query_marginals(ve, model, ["E"], {"A": "a0", "B": "b0", "C": "c1"}, priors)
        assert np.isnan(impossible["E"]).all()

        possible = _query_marginals(ve, model, ["E"], {"A": "a0", "B": "b1", "C": "c1"}, priors)
        np.testing.assert_allclose(possible["E"], [0.5, 0.5], atol=1e-6)


class TestImpossibleEvidenceWarning:
    """NaN for impossible evidence is accompanied by a warning naming the cause."""

    def _bn(self, model, a_values):
        bn = GeoBayesianNetwork(model)
        transform = Affine(1.0, 0, 0.0, 0, -1.0, 1.0)
        grids = {"A": np.array([a_values], dtype=float), "B": np.zeros((1, 3)), "D": np.ones((1, 3))}
        for node, grid in grids.items():
            bn.set_input(node, ArraySource(grid, crs="EPSG:32632", transform=transform))
            bn.set_discretization(node, [-0.5, 0.5, 1.5])
        return bn

    @pytest.mark.parametrize("use_table", [False, True])
    def test_infer_warns_and_returns_nan(self, impossible_root_model, use_table):
        bn = self._bn(impossible_root_model, [1, 0, 1])
        if use_table:
            bn.precompute(query=["C", "E"])
        with pytest.warns(UserWarning, match=r"2 pixel\(s\).*A='a1'.*prior probability of 0"):
            result = bn.infer(query=["C", "E"])
        for q in ["C", "E"]:
            probs = result.probabilities[q][0]
            assert np.isnan(probs[[0, 2]]).all()
            assert not np.isnan(probs[1]).any()

    def test_no_warning_when_evidence_is_possible(self, impossible_root_model, recwarn):
        bn = self._bn(impossible_root_model, [0, 0, 0])
        bn.infer(query=["C", "E"])
        bn.precompute(query=["C", "E"])     # the table holds NaN rows, but nothing observed them
        bn.infer(query=["C", "E"])
        bn.query_batch({"A": "a0", "B": "b0", "D": "d1"})
        assert not [w for w in recwarn if "probability zero" in str(w.message)]

    def test_query_batch_warns(self, impossible_root_model):
        bn = self._bn(impossible_root_model, [0, 0, 0])
        bn.precompute(query=["C", "E"])
        with pytest.warns(UserWarning, match=r"1 point\(s\).*A='a1'"):
            out = bn.query_batch({"A": ["a0", "a1"], "B": "b0", "D": "d1"})
        assert np.isnan(out["E"][1]).all() and not np.isnan(out["E"][0]).any()


class TestNonRootEvidenceThroughPublicAPI:
    """Roadmap 1.3: set_input() on a node with parents, end to end."""

    @staticmethod
    def _bn(model, grids: dict[str, list[list[float]]]) -> GeoBayesianNetwork:
        bn = GeoBayesianNetwork(model)
        transform = Affine(1.0, 0, 0.0, 0, -1.0, 1.0)
        for node, values in grids.items():
            arr = np.asarray(values, dtype=float)
            bn.set_input(node, ArraySource(arr, crs="EPSG:32632", transform=transform))
            bn.set_discretization(node, [-0.5, 0.5, 1.5])
        return bn

    def test_paths_agree(self, impossible_root_model, monkeypatch):
        """The per-combo loop and the joint table agree when evidence is non-root."""
        grids = {"A": [[0, 0, 1, 1]], "C": [[0, 1, 0, 1]]}
        loop = self._bn(impossible_root_model, grids).infer(query=["B", "E"])

        monkeypatch.setattr(inference_module, "_COMBO_LOOP_THRESHOLD", 0)
        table = self._bn(impossible_root_model, grids).infer(query=["B", "E"])

        for q in ["B", "E"]:
            np.testing.assert_allclose(
                loop.probabilities[q], table.probabilities[q], atol=1e-6, equal_nan=True
            )

    def test_precompute_agrees_with_pgmpy(self, impossible_root_model):
        """The precomputed table reproduces a direct VE query with non-root evidence."""
        bn = self._bn(impossible_root_model, {"A": [[0, 1]], "C": [[1, 0]]})
        pgmpy_path = bn.infer(query=["B"]).probabilities["B"]
        bn.precompute(query=["B"])
        table_path = bn.infer(query=["B"]).probabilities["B"]
        np.testing.assert_allclose(pgmpy_path, table_path, atol=1e-6, equal_nan=True)

        ve = VariableElimination(impossible_root_model)
        expected = ve.query(
            ["B"], evidence={"A": "a0", "C": "c1"}, show_progress=False
        ).values
        np.testing.assert_allclose(table_path[0, 0], expected, atol=1e-6)

    @pytest.mark.parametrize("use_table", [False, True])
    def test_contradictory_parent_child_evidence_is_nan(
        self, impossible_non_root_model, use_table
    ):
        """A = a0, B = b0, C = c1 has P(e) = 0, so every query node is NaN."""
        bn = self._bn(
            impossible_non_root_model,
            {"A": [[0, 0]], "B": [[0, 1]], "C": [[1, 1]]},
        )
        if use_table:
            bn.precompute(query=["E"])
        with pytest.warns(UserWarning, match="contradict each other"):
            probs = bn.infer(query=["E"]).probabilities["E"]
        assert np.isnan(probs[0, 0]).all()        # a0, b0, c1 -> impossible
        assert not np.isnan(probs[0, 1]).any()    # a0, b1, c1 -> fine
