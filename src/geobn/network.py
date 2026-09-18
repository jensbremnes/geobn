"""GeoBayesianNetwork — the primary user-facing class."""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork

from .discretize import DiscretizationSpec, discretize_array
from .grid import GridSpec, _pixel_size_m, align_to_grid
from .inference import (
    _query_marginals,
    _root_priors,
    build_conditional_table,
    run_inference,
    run_inference_from_table,
)
from .result import InferenceResult
from .sources._base import DataSource

_log = logging.getLogger(__name__)

# Bump when the __metadata__ layout written by save_precomputed changes.
_TABLE_FORMAT_VERSION = 2


def _geobn_version() -> str:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        return version("geobn")
    except PackageNotFoundError:
        return "unknown"


def _model_hash(model: DiscreteBayesianNetwork) -> str:
    """SHA-256 of the BN's structure, state names and CPD values.

    Values are rounded to 10 decimals so a BIF write/read round-trip hashes
    the same.
    """
    cpds = []
    for cpd in sorted(model.get_cpds(), key=lambda c: c.variable):
        cpds.append({
            "variables": list(cpd.variables),
            "state_names": {v: list(cpd.state_names[v]) for v in cpd.variables},
            "values": np.round(np.asarray(cpd.values, dtype=np.float64), 10).tolist(),
        })
    canonical = {
        "nodes": sorted(model.nodes()),
        "edges": sorted([list(e) for e in model.edges()]),
        "cpds": cpds,
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GeoBayesianNetwork:
    """A Bayesian network wired to geographic data sources.

    Typical usage::

        bn = geobn.load("model.bif")
        bn.set_input("slope",    geobn.RasterSource("slope.tif"))
        bn.set_input("rainfall", geobn.ConstantSource(50.0))
        bn.set_discretization("slope",    [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])
        result = bn.infer(query=["fire_risk"])
        result.to_geotiff("output/")

    Real-time / repeated inference
    --------------------------------
    When only a subset of inputs change between calls (e.g. static terrain,
    changing weather), use :meth:`freeze` to cache static node arrays::

        bn.freeze("slope_angle", "aspect")          # terrain is static
        result = bn.infer(query=["avalanche_risk"]) # first call: fetches & caches terrain

        # Subsequent calls re-process only weather inputs:
        bn.set_input("recent_snow", geobn.ConstantSource(35.0))
        result = bn.infer(query=["avalanche_risk"]) # terrain reused from cache

    For maximum throughput, pre-run all evidence combinations once::

        bn.precompute(query=["avalanche_risk"])      # one-time cost: all combos
        result = bn.infer(query=["avalanche_risk"])  # O(H×W) numpy indexing, no pgmpy
    """

    def __init__(self, model: DiscreteBayesianNetwork) -> None:
        """
        Parameters
        ----------
        model:
            A fitted ``pgmpy.models.DiscreteBayesianNetwork``.
        """
        if not isinstance(model, DiscreteBayesianNetwork):
            raise TypeError(
                f"Expected DiscreteBayesianNetwork, got {type(model).__name__}"
            )
        self._model = model
        self._inputs: dict[str, DataSource] = {}
        self._discretizations: dict[str, DiscretizationSpec] = {}
        self._grid: GridSpec | None = None

        # ── Real-time optimisation state ─────────────────────────────────────
        # Tier 1 — frozen input cache
        self._frozen_nodes: set[str] = set()
        self._frozen_cache: dict[str, np.ndarray] = {}   # node → (H,W) int16 array
        self._cached_ref_grid: GridSpec | None = None
        self._cached_ve: Any | None = None               # VariableElimination instance
        # Tier 2 — precomputed inference table
        self._inference_table: dict[str, np.ndarray] = {}
        self._evidence_nodes: list[str] = []
        self._query_nodes: list[str] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_input(self, node: str, source: DataSource) -> None:
        """Attach a data source to a BN evidence node.

        Parameters
        ----------
        node:
            Name of a root node (no parents) in the BN.
        source:
            Any :class:`~geobn.sources.DataSource` subclass.
        """
        self._validate_node_exists(node)
        self._validate_is_root(node)
        self._inputs[node] = source
        # If this node was frozen and cached, the cached array is now stale
        if self._frozen_cache.pop(node, None) is not None:
            self._invalidate_table()
        _log.info("Input: '%s' ← %s", node, type(source).__name__)

    def set_discretization(
        self,
        node: str,
        breakpoints: list[float],
        labels: list[str] | None = None,
        out_of_range: str = "clip",
    ) -> None:
        """Define how continuous values for *node* are mapped to BN states.

        Parameters
        ----------
        node:
            Node name (must match an input node already registered via
            :meth:`set_input`).
        breakpoints:
            Monotonically increasing list of ``len(labels) + 1`` boundary
            values.  The interior values define the bin edges; the first and
            last define the valid range ``[first, last]`` used by
            *out_of_range*.
        labels:
            State names that **exactly** match the state names in the BN.
            If omitted, state names are read from the BN node in their
            definition order — the breakpoints must then produce exactly
            as many bins as the node has states.
        out_of_range:
            What to do with values outside ``[breakpoints[0], breakpoints[-1]]``:

            - ``"clip"`` (default): assign them to the first or last state.
            - ``"nan"``: treat them as NoData, so the pixel (or point) gets
              NaN probabilities.
        """
        self._validate_node_exists(node)
        if labels is None:
            cpd = self._model.get_cpds(node)
            labels = list(cpd.state_names[node])
        spec = DiscretizationSpec(
            breakpoints=list(breakpoints),
            labels=list(labels),
            out_of_range=out_of_range,
        )
        self._validate_labels_match_bn(node, spec.labels)
        self._discretizations[node] = spec
        # If this node was frozen and cached, the cached array used the old spec
        if self._frozen_cache.pop(node, None) is not None:
            self._invalidate_table()
        _log.info("Discretization: '%s' → %d bins %s", node, len(labels), labels)

    def set_grid(
        self,
        crs: str,
        resolution: float,
        extent: tuple[float, float, float, float],
    ) -> None:
        """Override the reference grid instead of deriving it from the first input.

        Parameters
        ----------
        crs:
            Target CRS as EPSG string (e.g. "EPSG:32632") or WKT.
        resolution:
            Pixel size in CRS units.
        extent:
            (xmin, ymin, xmax, ymax) in CRS units.
        """
        self._grid = GridSpec.from_params(crs, resolution, extent)
        self._cached_ref_grid = None
        self._frozen_cache.clear()
        self._invalidate_table()
        H, W = self._grid.shape
        _log.info("Grid set: %s, resolution=%g, shape=%d×%d", crs, resolution, H, W)

    def fetch_raw(self, source: DataSource) -> np.ndarray:
        """Fetch a data source using the BN's grid and return a plain numpy array.

        Useful when you need the raw values to derive additional inputs — for
        example, fetching a DEM to compute slope and aspect before registering
        them via :meth:`set_input`.  Requires :meth:`set_grid` to be called
        first.

        Parameters
        ----------
        source:
            Any :class:`~geobn.sources.DataSource` to fetch.  The source is
            not registered as an input.

        Returns
        -------
        np.ndarray
            Float32 array of shape (H, W), aligned to the BN's grid.
            NaN where the source has no data.
        """
        if self._grid is None:
            raise RuntimeError(
                "No grid configured.  Call bn.set_grid(crs, resolution, extent) first."
            )
        data = source.fetch(grid=self._grid)
        return align_to_grid(data, self._grid)

    # ------------------------------------------------------------------
    # Real-time optimisation
    # ------------------------------------------------------------------

    def freeze(self, *node_names: str) -> None:
        """Mark one or more input nodes as static.

        On the first :meth:`infer` call after freezing, each frozen node is
        fetched, aligned to the grid, and discretised normally; the resulting
        integer index array is then cached in memory.  On all subsequent calls
        the cached array is reused, skipping fetch, alignment, and
        discretisation for those nodes.

        Calling :meth:`freeze` with a different set of nodes invalidates any
        previously cached data.

        Parameters
        ----------
        *node_names:
            Names of input nodes whose data will not change between
            :meth:`infer` calls.
        """
        for name in node_names:
            self._validate_node_exists(name)
        new_frozen = set(node_names)
        if new_frozen != self._frozen_nodes:
            self._frozen_nodes = new_frozen
            self.clear_cache()
        _log.info("Freezing %d node(s): %s", len(node_names), list(node_names))

    def clear_cache(self) -> None:
        """Invalidate all cached discrete arrays and the inference table.

        Call this if a frozen input actually changes (e.g. you replaced the
        terrain source), or after calling :meth:`freeze` with a different set
        of nodes.  The next :meth:`infer` call will re-fetch and re-cache all
        frozen nodes.
        """
        self._frozen_cache.clear()
        self._cached_ref_grid = None
        self._cached_ve = None
        self._invalidate_table()
        _log.debug("Cache cleared")

    def _invalidate_table(self) -> None:
        """Drop the precomputed inference table and its node bookkeeping."""
        self._inference_table.clear()
        self._evidence_nodes = []
        self._query_nodes = []

    def precompute(self, query: list[str]) -> None:
        """Pre-run all evidence-state combinations and store a lookup table.

        After :meth:`precompute`, subsequent :meth:`infer` calls for the same
        *query* nodes bypass pgmpy entirely: probabilities are fetched from the
        table via numpy fancy indexing — O(H×W) rather than O(n_unique_combos)
        pgmpy queries.

        One-time cost: a single pgmpy joint query per query node (the joint
        P(query, e_1, ..., e_k) is normalised into the conditional table).
        Only when the table would exceed the in-memory size bound does this
        fall back to one query per evidence combination.

        Parameters
        ----------
        query:
            BN node names to precompute posteriors for.  Must match the
            *query* passed to :meth:`infer` for the table path to activate.

        Notes
        -----
        State names are read directly from the BN's CPDs, so
        :meth:`set_discretization` is not required before calling
        :meth:`precompute`.
        """
        for node in query:
            self._validate_node_exists(node)

        from pgmpy.inference import VariableElimination  # noqa: PLC0415

        node_order = list(self._inputs.keys())
        state_names_per_node = {
            n: list(self._model.get_cpds(n).state_names[n]) for n in node_order
        }
        n_states_per_node = [len(state_names_per_node[n]) for n in node_order]

        query_state_names: dict[str, list[str]] = {}
        for qnode in query:
            cpd = self._model.get_cpds(qnode)
            query_state_names[qnode] = list(cpd.state_names[qnode])

        if self._cached_ve is None:
            self._cached_ve = VariableElimination(self._model)
        ve = self._cached_ve

        n_total = 1
        for k in n_states_per_node:
            n_total *= k
        _log.info("Precomputing inference table: %d evidence combination(s) ...", n_total)

        # Fast path: one joint VE query per query node covers every evidence
        # combination at once.  Returns None only when the table exceeds the
        # in-memory size bound — then fall back to the per-combination loop.
        tables = build_conditional_table(self._model, node_order, query, ve=ve)

        if tables is None:
            _log.warning(
                "Conditional table too large for the joint-query strategy — "
                "enumerating %d combination(s) individually (this may be slow)",
                n_total,
            )
            tables = {}
            for qnode in query:
                n_q = len(query_state_names[qnode])
                tables[qnode] = np.zeros(n_states_per_node + [n_q], dtype=np.float32)

            root_priors = _root_priors(self._model)
            for idx_combo in itertools.product(*[range(k) for k in n_states_per_node]):
                evidence = {
                    node_order[i]: state_names_per_node[node_order[i]][idx_combo[i]]
                    for i in range(len(node_order))
                }
                marginals = _query_marginals(ve, self._model, query, evidence, root_priors)
                for qnode in tables:
                    tables[qnode][idx_combo] = marginals[qnode]

        self._inference_table = tables
        self._evidence_nodes = node_order
        self._query_nodes = list(query)
        _log.info("Precompute done.  Table shape: %s", next(iter(tables.values())).shape)

    def save_precomputed(self, path: str | Path) -> None:
        """Serialize the precomputed lookup table to a portable ``.npz`` file.

        The file can be loaded on any machine with
        :meth:`load_precomputed` — no pgmpy is required at load time.

        Parameters
        ----------
        path:
            Destination path.  A ``.npz`` extension is appended automatically
            if not already present (numpy behaviour).

        Raises
        ------
        RuntimeError
            If :meth:`precompute` has not been called yet.
        """
        if not self._inference_table:
            raise RuntimeError(
                "No precomputed table. Call precompute() first."
            )
        # __metadata__ records what the table was built from, so load_precomputed
        # can refuse a table that doesn't fit the current BN.  Example for a BN
        # with slope + rainfall → fire_risk:
        #   {
        #       "format_version": 2,
        #       "geobn_version":  "0.1.1",
        #       "evidence_nodes": ["slope", "rainfall"],  # input axes of the table
        #       "query_nodes":    ["fire_risk"],          # stored posteriors
        #       "model_hash":     "<sha256 of structure, states and CPDs>",
        #       "state_names":    {"slope": ["flat", ...], ..., "fire_risk": [...]},
        #       "discretizations": {"slope": {"breakpoints": [...], "labels": [...],
        #                                     "out_of_range": "clip"},
        #                           "rainfall": null}      # null: not set at save
        #   }
        # The "fire_risk" array then has shape (n_slope_states, n_rainfall_states,
        # n_fire_risk_states), e.g. (3, 3, 3), with axes in BN state order.
        discretizations: dict[str, dict[str, Any] | None] = {}
        for n in self._evidence_nodes:
            spec = self._discretizations.get(n)
            discretizations[n] = None if spec is None else {
                "breakpoints": [float(b) for b in spec.breakpoints],
                "labels": list(spec.labels),
                "out_of_range": spec.out_of_range,
            }
        metadata = {
            "format_version": _TABLE_FORMAT_VERSION,
            "geobn_version": _geobn_version(),
            "evidence_nodes": self._evidence_nodes,
            "query_nodes": self._query_nodes,
            "model_hash": _model_hash(self._model),
            "state_names": {
                n: self._bn_state_names(n)
                for n in [*self._evidence_nodes, *self._query_nodes]
            },
            "discretizations": discretizations,
        }
        arrays = dict(self._inference_table)
        arrays["__metadata__"] = np.array([json.dumps(metadata)])
        np.savez_compressed(path, **arrays)
        _log.info("Saved precomputed table to '%s'", path)

    def load_precomputed(self, path: str | Path) -> None:
        """Load a precomputed lookup table saved with :meth:`save_precomputed`.

        After loading, :meth:`infer` uses the table path (O(H×W) numpy
        indexing) without calling pgmpy.

        The file records the model it was built from, and loading checks it
        against the current BN:

        - The table's evidence nodes must be the current inputs.  If they were
          registered in a different order, the table axes are reordered.
        - The BN state names and a hash of the model (structure, states and
          CPD values) must match.
        - Discretizations saved with the table are restored for inputs that
          have none set.  An input whose discretization is already set must
          match the saved one.

        Files written by older geobn versions carry none of this information.
        They still load, with a warning, and only the evidence nodes and array
        shapes are checked.

        Parameters
        ----------
        path:
            Path to the ``.npz`` file written by :meth:`save_precomputed`.

        Raises
        ------
        FileNotFoundError
            If neither *path* nor *path* + ``.npz`` exists.
        ValueError
            If the table's evidence nodes, state names, model hash,
            discretizations or array shapes do not match the current BN
            configuration.
        """
        path = Path(path)
        if not path.exists():
            npz_path = path.with_suffix(".npz")
            if npz_path.exists():
                path = npz_path
            else:
                raise FileNotFoundError(
                    f"Precomputed table file not found: '{path}'"
                )

        data = np.load(path, allow_pickle=False)

        if "__metadata__" not in data:
            raise ValueError(
                f"File '{path}' is missing the '__metadata__' key.  "
                "Was it saved with save_precomputed()?"
            )
        metadata = json.loads(str(data["__metadata__"][0]))
        node_order: list[str] = metadata["evidence_nodes"]
        query_nodes: list[str] = metadata["query_nodes"]
        legacy = "format_version" not in metadata
        if legacy:
            warnings.warn(
                f"'{path}' was saved by an older geobn and does not record the "
                "model it was built from, so it cannot be fully validated.  "
                "Re-run precompute() and save_precomputed() to upgrade it.",
                UserWarning,
                stacklevel=2,
            )

        # The evidence nodes must be the current inputs; their order may differ
        current_order = list(self._inputs.keys())
        if sorted(node_order) != sorted(current_order):
            raise ValueError(
                f"Evidence node mismatch: table has {node_order}, "
                f"current inputs are {current_order}.  "
                "Register the same inputs or re-run precompute()."
            )

        # Validate every query node exists in the BN
        for n in query_nodes:
            self._validate_node_exists(n)

        if not legacy:
            for n, saved in metadata["state_names"].items():
                self._validate_node_exists(n)
                if saved != self._bn_state_names(n):
                    raise ValueError(
                        f"State names for '{n}' differ: table has {saved}, "
                        f"the BN has {self._bn_state_names(n)}.  Re-run precompute()."
                    )
            if metadata["model_hash"] != _model_hash(self._model):
                raise ValueError(
                    f"The table in '{path}' was built from a different model "
                    "(structure or CPD values differ).  Re-run precompute()."
                )
            self._restore_discretizations(metadata["discretizations"])

        # Validate array shapes match current discretizations
        missing = [n for n in node_order if n not in self._discretizations]
        if missing:
            raise ValueError(
                f"No discretization set for input node(s) {missing}.  "
                "Call set_discretization() for every input before load_precomputed()."
            )
        expected_n_states = [
            len(self._discretizations[n].labels) for n in node_order
        ]
        for qnode in query_nodes:
            if qnode not in data:
                raise ValueError(
                    f"Query node '{qnode}' not found in the file.  "
                    "The file may be corrupt or from an incompatible save."
                )
            arr = data[qnode]
            actual = list(arr.shape[:-1])
            if actual != expected_n_states:
                raise ValueError(
                    f"Shape mismatch for '{qnode}': table evidence axes {actual} "
                    f"do not match current discretization n_states {expected_n_states}.  "
                    "Ensure discretization specs match those used when the table was saved."
                )

        # Reorder the evidence axes to the current input order; query axis stays last
        axes = [node_order.index(n) for n in current_order] + [len(node_order)]
        self._inference_table = {
            qnode: np.ascontiguousarray(np.transpose(data[qnode], axes), dtype=np.float32)
            for qnode in query_nodes
        }
        self._evidence_nodes = current_order
        self._query_nodes = query_nodes
        _log.info(
            "Loaded precomputed table from '%s': query=%s, evidence=%s",
            path, query_nodes, current_order,
        )

    def _restore_discretizations(self, saved: dict[str, dict[str, Any] | None]) -> None:
        """Adopt saved discretizations for unset inputs; raise if a set one differs."""
        for node, entry in saved.items():
            if entry is None:
                continue
            spec = DiscretizationSpec(
                breakpoints=list(entry["breakpoints"]),
                labels=list(entry["labels"]),
                out_of_range=entry["out_of_range"],
            )
            current = self._discretizations.get(node)
            if current is None:
                self._validate_labels_match_bn(node, spec.labels)
                self._discretizations[node] = spec
                _log.info("Discretization for '%s' restored from table", node)
            elif (
                [float(b) for b in current.breakpoints] != spec.breakpoints
                or current.labels != spec.labels
                or current.out_of_range != spec.out_of_range
            ):
                raise ValueError(
                    f"Discretization for '{node}' differs from the one saved with "
                    f"the table (saved: breakpoints={spec.breakpoints}, "
                    f"labels={spec.labels}, out_of_range={spec.out_of_range!r}).  "
                    "Remove the set_discretization() call to use the saved one, "
                    "or re-run precompute() and save_precomputed()."
                )

    # ------------------------------------------------------------------
    # Point queries
    # ------------------------------------------------------------------

    def query_point(
        self,
        evidence: dict[str, float | str],
        query: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Posterior distributions for a single evidence point.

        After :meth:`precompute` (or :meth:`load_precomputed`), this is a pure
        numpy lookup in the precomputed table: no pgmpy inference runs, and
        no grid or data sources are involved.

        Parameters
        ----------
        evidence:
            One value per input node (every node passed to :meth:`set_input`).
            Values may be state names (``"steep"``) or numbers, which are
            discretized with the node's :meth:`set_discretization` spec.
            A NaN value gives NaN probabilities, as in :meth:`infer`.
        query:
            Query nodes to return.  Defaults to the nodes passed to
            :meth:`precompute`; must be a subset of them.

        Returns
        -------
        dict
            ``{query_node: {state_name: probability}}``.

        Raises
        ------
        RuntimeError
            If no precomputed table is available.
        ValueError
            On missing evidence, unknown state names, sequence-valued
            evidence, or a query node that was not precomputed.

        Example
        -------
        >>> bn.precompute(query=["fire_risk"])
        >>> bn.query_point({"slope": 35.0, "rainfall": "high"})
        {'fire_risk': {'low': 0.2, 'medium': 0.3, 'high': 0.5}}
        """
        for node, value in evidence.items():
            if not isinstance(value, str) and np.ndim(value) > 0:
                raise ValueError(
                    f"Evidence for '{node}' must be a single value; "
                    "use query_batch() for multiple points."
                )
        batch = self.query_batch(evidence, query)
        return {
            qnode: {
                state: float(p)
                for state, p in zip(self._model.get_cpds(qnode).state_names[qnode], probs[0])
            }
            for qnode, probs in batch.items()
        }

    def query_batch(
        self,
        evidence: dict[str, Any],
        query: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Posterior distributions for K evidence points at once.

        Like :meth:`query_point`, but vectorized: a pure numpy lookup in the
        precomputed table.

        Parameters
        ----------
        evidence:
            One entry per input node.  An entry is either a sequence of K
            values (list, tuple or array) or a single value that applies to
            all K points.  Values may be numbers (discretized with the node's
            spec) or state names, but not both in one sequence.  NaN values
            give a NaN row for that point.
        query:
            Query nodes to return.  Defaults to the nodes passed to
            :meth:`precompute`; must be a subset of them.

        Returns
        -------
        dict
            ``{query_node: (K, n_states) float32 array}`` with state order
            matching the BN definition.

        Raises
        ------
        RuntimeError
            If no precomputed table is available.
        ValueError
            On missing evidence, unknown state names, sequences of different
            lengths, or a query node that was not precomputed.
        """
        if not self._inference_table:
            raise RuntimeError(
                "No precomputed table.  Call precompute(query) or "
                "load_precomputed(path) first."
            )
        query = self._validate_point_query(query)
        missing = [n for n in self._evidence_nodes if n not in evidence]
        if missing:
            raise ValueError(f"Missing evidence for node(s): {missing}")

        values = {n: self._as_evidence_array(n, evidence[n]) for n in self._evidence_nodes}
        lengths = {len(v) for v in values.values() if v.ndim == 1}
        if len(lengths) > 1:
            raise ValueError(f"Evidence sequences have mismatched lengths: {sorted(lengths)}")
        k = lengths.pop() if lengths else 1

        index_arrays: list[np.ndarray] = []
        invalid = np.zeros(k, dtype=bool)
        for node in self._evidence_nodes:
            idx = np.broadcast_to(self._evidence_state_indices(node, values[node]), (k,))
            invalid |= idx < 0
            index_arrays.append(np.clip(idx, 0, None))

        out: dict[str, np.ndarray] = {}
        for qnode in query:
            probs = self._inference_table[qnode][tuple(index_arrays)].astype(np.float32)
            probs[invalid] = np.nan
            out[qnode] = probs
        return out

    @staticmethod
    def _as_evidence_array(node: str, value: Any) -> np.ndarray:
        """Convert one evidence entry to a 0-d or 1-d array of numbers or state names."""
        # Lists go through object dtype so numpy doesn't coerce [5.0, "steep"] to strings
        arr = np.array(value, dtype=object) if isinstance(value, (list, tuple)) else np.asarray(value)
        if arr.ndim > 1:
            raise ValueError(f"Evidence for '{node}' must be a single value or a 1-D sequence.")
        if arr.dtype.kind in "iufb":
            return arr.astype(float)
        if arr.dtype.kind == "U":
            return arr
        if arr.dtype.kind == "O":
            if all(isinstance(v, str) for v in arr.flat):
                return arr.astype(str)
            if all(isinstance(v, (int, float, np.number)) for v in arr.flat):
                return arr.astype(float)
        raise ValueError(
            f"Evidence for '{node}' must be numbers or state names, not a mix of both."
        )

    def _evidence_state_indices(self, node: str, values: np.ndarray) -> np.ndarray:
        """Map evidence values (numbers or state names) to state indices; NaN → -1."""
        if values.dtype.kind == "U":
            state_names = list(self._model.get_cpds(node).state_names[node])
            unknown = sorted({str(v) for v in values.flat} - set(state_names))
            if unknown:
                raise ValueError(
                    f"Unknown state(s) {unknown} for node '{node}'.  "
                    f"Expected one of {state_names}."
                )
            lookup = {s: i for i, s in enumerate(state_names)}
            return np.vectorize(lookup.__getitem__, otypes=[int])(values)
        spec = self._discretizations.get(node)
        if spec is None:
            raise ValueError(
                f"Numeric evidence for '{node}' requires a discretization.  "
                f"Call set_discretization('{node}', breakpoints, labels) or pass "
                "state names instead."
            )
        flat = self._to_bn_state_order(node, discretize_array(values.reshape(1, -1), spec))[0]
        return flat.reshape(values.shape).astype(int)

    def _bn_state_names(self, node: str) -> list[str]:
        return list(self._model.get_cpds(node).state_names[node])

    def _to_bn_state_order(self, node: str, idx: np.ndarray) -> np.ndarray:
        """Map indices into the discretization's labels to indices into the BN's states.

        Labels may be given in any order, but the precomputed table axes (and the
        state names passed to pgmpy) follow the BN's state order.  -1 stays -1.
        """
        labels = self._discretizations[node].labels
        bn_states = self._bn_state_names(node)
        if labels == bn_states:
            return idx
        perm = np.array([bn_states.index(label) for label in labels], dtype=idx.dtype)
        return np.where(idx >= 0, perm[np.clip(idx, 0, None)], idx).astype(idx.dtype)

    def _validate_point_query(self, query: list[str] | None) -> list[str]:
        if query is None:
            return list(self._query_nodes)
        unknown = [q for q in query if q not in self._query_nodes]
        if unknown:
            raise ValueError(
                f"Query node(s) {unknown} were not precomputed.  "
                f"Precomputed query nodes: {self._query_nodes}."
            )
        return list(query)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, query: list[str]) -> InferenceResult:
        """Run pixel-wise Bayesian inference and return probability rasters.

        Parameters
        ----------
        query:
            List of BN node names whose posterior distributions are requested.
            These nodes do not need to be root nodes.

        Returns
        -------
        InferenceResult
            Contains per-pixel probability arrays for each query node plus
            Shannon entropy.  Write to disk with ``.to_geotiff()`` or convert
            to xarray with ``.to_xarray()``.

        Notes
        -----
        If :meth:`precompute` has been called with the same *query*, this
        method uses numpy fancy indexing instead of pgmpy queries.  If
        :meth:`freeze` has been called, cached discrete arrays are reused for
        frozen nodes.
        """
        if not self._inputs:
            raise RuntimeError("No inputs registered.  Call set_input() first.")

        for node in query:
            self._validate_node_exists(node)

        # ── 1. Determine the reference grid ───────────────────────────
        if self._grid is not None:
            ref_grid = self._grid
            pre_fetched: dict[str, Any] = {}
        elif self._cached_ref_grid is not None:
            # Grid already established from a previous call with frozen nodes
            ref_grid = self._cached_ref_grid
            pre_fetched = {}
        else:
            # Fetch every self-contained (non-grid-aware) source up front so we
            # can inspect its CRS and resolution.  Grid-aware sources cannot be
            # fetched yet — they need the bbox, which we don't know yet.
            pre_fetched = {}
            candidate_grids: list[tuple[float, GridSpec, str]] = []  # (pixel_size_m, grid, node)

            for node, source in self._inputs.items():
                if source.requires_grid:
                    continue  # needs bbox first — skip for now
                data = source.fetch(grid=None)
                pre_fetched[node] = data
                if data.crs is None:
                    continue  # ConstantSource or similar — no spatial info
                grid_candidate = GridSpec.from_raster_data(data)
                # Compare in metres: CRS units differ between sources (degrees vs metres)
                candidate_grids.append((_pixel_size_m(grid_candidate), grid_candidate, node))

            if not candidate_grids:
                raise ValueError(
                    "Could not determine a reference grid automatically.  "
                    "All registered sources require a grid bbox before they "
                    "can fetch data (requires_grid=True).  "
                    "Call bn.set_grid(crs, resolution, extent) explicitly."
                )

            # Pick the source with the finest (smallest) pixel size so that
            # high-resolution sources are never downsampled unnecessarily.
            smallest_pixel_m, ref_grid, ref_node = min(candidate_grids, key=lambda t: t[0])
            _log.info(
                "Auto-selected reference grid from '%s' (finest resolution, ~%.3g m pixels)",
                ref_node, smallest_pixel_m,
            )

        _log.info(
            "Reference grid: %s, shape=%d×%d, resolution=%g",
            ref_grid.crs, ref_grid.shape[0], ref_grid.shape[1], ref_grid.transform.a,
        )

        # ── 2. Validate discretizations are present for all inputs ─────
        for node in self._inputs:
            if node not in self._discretizations:
                raise ValueError(
                    f"No discretization set for input node '{node}'.  "
                    f"Call bn.set_discretization('{node}', breakpoints, labels)."
                )

        # ── 3. Fetch, align, and discretize inputs ─────────────────────
        # Frozen nodes with a cached discrete array skip all I/O and compute.
        evidence_state_grids: dict[str, np.ndarray] = {}
        evidence_state_names: dict[str, list[str]] = {}
        nodata_mask = np.zeros(ref_grid.shape, dtype=bool)

        for node, source in self._inputs.items():
            spec = self._discretizations[node]

            if node in self._frozen_nodes and node in self._frozen_cache:
                # Fast path: reuse cached discrete index array
                _log.info("Frozen cache hit: '%s'", node)
                idx = self._frozen_cache[node]
            else:
                _log.info("Fetching '%s' from %s", node, type(source).__name__)
                data = (
                    pre_fetched[node]
                    if node in pre_fetched
                    else source.fetch(grid=ref_grid)
                )
                arr = align_to_grid(data, ref_grid)
                idx = self._to_bn_state_order(node, discretize_array(arr, spec))

                if node in self._frozen_nodes:
                    # Cache discrete array; also cache the grid so the next call
                    # can skip the first-node fetch that derives the grid.
                    self._frozen_cache[node] = idx
                    if self._cached_ref_grid is None:
                        self._cached_ref_grid = ref_grid

            nodata_mask |= idx < 0
            evidence_state_grids[node] = idx
            evidence_state_names[node] = self._bn_state_names(node)

        # ── 4. Collect query node state names from the BN ──────────────
        query_state_names: dict[str, list[str]] = {}
        for node in query:
            cpd = self._model.get_cpds(node)
            query_state_names[node] = list(cpd.state_names[node])

        # ── 5. Run inference ───────────────────────────────────────────
        if (
            self._inference_table
            and sorted(query) == sorted(self._query_nodes)
            and list(self._inputs.keys()) == self._evidence_nodes
        ):
            # Tier-2 fast path: pure numpy table lookup, no pgmpy per call
            _log.info("Using precomputed table (zero pgmpy calls)")
            probabilities = run_inference_from_table(
                table=self._inference_table,
                node_order=self._evidence_nodes,
                evidence_state_grids=evidence_state_grids,
                nodata_mask=nodata_mask,
            )
        else:
            # Normal path (Tier-1 or uncached): pgmpy VE with cached engine
            if self._cached_ve is None:
                from pgmpy.inference import VariableElimination  # noqa: PLC0415

                self._cached_ve = VariableElimination(self._model)

            probabilities = run_inference(
                model=self._model,
                evidence_state_grids=evidence_state_grids,
                evidence_state_names=evidence_state_names,
                query_nodes=query,
                query_state_names=query_state_names,
                nodata_mask=nodata_mask,
                ve=self._cached_ve,
            )

        n_valid = int((~nodata_mask).sum())
        _log.info(
            "Inference complete: %d×%d pixels, %d valid",
            ref_grid.shape[0], ref_grid.shape[1], n_valid,
        )

        return InferenceResult(
            probabilities=probabilities,
            state_names=query_state_names,
            crs=ref_grid.crs,
            transform=ref_grid.transform,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_node_exists(self, node: str) -> None:
        if node not in self._model.nodes():
            raise ValueError(
                f"Node '{node}' does not exist in the BN.  "
                f"Available nodes: {sorted(self._model.nodes())}"
            )

    def _validate_is_root(self, node: str) -> None:
        parents = list(self._model.predecessors(node))
        if parents:
            raise ValueError(
                f"Node '{node}' has parents {parents} and is not a root node.  "
                f"Only root nodes (no parents) can be used as inputs."
            )

    def _validate_labels_match_bn(self, node: str, labels: list[str]) -> None:
        cpd = self._model.get_cpds(node)
        bn_states = list(cpd.state_names[node])
        if sorted(labels) != sorted(bn_states):
            raise ValueError(
                f"Discretization labels {labels} for node '{node}' do not "
                f"match the BN state names {bn_states}.  "
                f"Labels must exactly match (order-independent)."
            )


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------


def load(path: str | Path) -> GeoBayesianNetwork:
    """Load a Bayesian network from a BIF file.

    Parameters
    ----------
    path:
        Path to a ``.bif`` file.

    Returns
    -------
    GeoBayesianNetwork
        Ready to accept inputs via :meth:`~GeoBayesianNetwork.set_input`.
    """
    from pgmpy.readwrite import BIFReader  # noqa: PLC0415

    reader = BIFReader(str(Path(path)))
    model = reader.get_model()  # returns DiscreteBayesianNetwork in pgmpy >=1.0
    _log.info("Loaded BN from '%s': %d nodes", Path(path).name, len(model.nodes()))
    return GeoBayesianNetwork(model)
