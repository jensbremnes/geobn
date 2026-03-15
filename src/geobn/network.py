"""GeoBayesianNetwork — the primary user-facing class."""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork

from .discretize import DiscretizationSpec, discretize_array
from .grid import GridSpec, align_to_grid
from .inference import run_inference, run_inference_from_table
from .result import InferenceResult
from .sources._base import DataSource


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
            self._inference_table.clear()
            self._evidence_nodes = []
            self._query_nodes = []
        _log.info("Input: '%s' ← %s", node, type(source).__name__)

    def set_discretization(
        self,
        node: str,
        breakpoints: list[float],
        labels: list[str] | None = None,
    ) -> None:
        """Define how continuous values for *node* are mapped to BN states.

        Parameters
        ----------
        node:
            Node name (must match an input node already registered via
            :meth:`set_input`).
        breakpoints:
            Monotonically increasing list of ``len(labels) + 1`` boundary
            values.  The first and last are the documented range limits; the
            interior values define the bin edges.
        labels:
            State names that **exactly** match the state names in the BN.
            If omitted, state names are read from the BN node in their
            definition order — the breakpoints must then produce exactly
            as many bins as the node has states.
        """
        self._validate_node_exists(node)
        if labels is None:
            cpd = self._model.get_cpds(node)
            labels = list(cpd.state_names[node])
        spec = DiscretizationSpec(breakpoints=list(breakpoints), labels=list(labels))
        self._validate_labels_match_bn(node, spec.labels)
        self._discretizations[node] = spec
        # If this node was frozen and cached, the cached array used the old spec
        if self._frozen_cache.pop(node, None) is not None:
            self._inference_table.clear()
            self._evidence_nodes = []
            self._query_nodes = []
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
        self._inference_table.clear()
        self._evidence_nodes = []
        self._query_nodes = []
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
        self._inference_table.clear()
        self._evidence_nodes = []
        self._query_nodes = []
        _log.debug("Cache cleared")

    def precompute(self, query: list[str]) -> None:
        """Pre-run all evidence-state combinations and store a lookup table.

        After :meth:`precompute`, subsequent :meth:`infer` calls for the same
        *query* nodes bypass pgmpy entirely: probabilities are fetched from the
        table via numpy fancy indexing — O(H×W) rather than O(n_unique_combos)
        pgmpy queries.

        One-time cost: ``∏ n_states_i`` pgmpy queries.  For the Lyngen Alps BN
        (3×2×3×3 state space) this is 54 queries, typically completing in well
        under a second.

        Parameters
        ----------
        query:
            BN node names to precompute posteriors for.  Must match the
            *query* passed to :meth:`infer` for the table path to activate.

        Notes
        -----
        All inputs must have discretizations configured via
        :meth:`set_discretization` before calling :meth:`precompute`.
        """
        for node in query:
            self._validate_node_exists(node)
        for node in self._inputs:
            if node not in self._discretizations:
                raise RuntimeError(
                    f"No discretization set for '{node}'.  "
                    f"Call set_discretization() for all inputs before precompute()."
                )

        from pgmpy.inference import VariableElimination  # noqa: PLC0415

        node_order = list(self._inputs.keys())
        state_names_per_node = {n: self._discretizations[n].labels for n in node_order}
        n_states_per_node = [len(state_names_per_node[n]) for n in node_order]

        query_state_names: dict[str, list[str]] = {}
        for qnode in query:
            cpd = self._model.get_cpds(qnode)
            query_state_names[qnode] = list(cpd.state_names[qnode])

        if self._cached_ve is None:
            self._cached_ve = VariableElimination(self._model)
        ve = self._cached_ve

        # Allocate tables: shape (*n_states_per_node, n_q_states) for each query node
        tables: dict[str, np.ndarray] = {}
        for qnode in query:
            n_q = len(query_state_names[qnode])
            tables[qnode] = np.zeros(n_states_per_node + [n_q], dtype=np.float32)

        n_total = 1
        for k in n_states_per_node:
            n_total *= k
        _log.info("Precomputing inference table: %d evidence combination(s) ...", n_total)

        for idx_combo in itertools.product(*[range(k) for k in n_states_per_node]):
            evidence = {
                node_order[i]: state_names_per_node[node_order[i]][idx_combo[i]]
                for i in range(len(node_order))
            }
            for qnode in query:
                factor = ve.query([qnode], evidence=evidence, show_progress=False)
                tables[qnode][idx_combo] = factor.values.astype(np.float32)

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
        # __metadata__ encodes which nodes form the table axes and which are
        # query outputs.  Example for a BN with slope + rainfall → fire_risk:
        #   {
        #       "evidence_nodes": ["slope", "rainfall"],  # input axes of the table
        #       "query_nodes":    ["fire_risk"]           # stored posteriors
        #   }
        # The "fire_risk" array then has shape (n_slope_states, n_rainfall_states,
        # n_fire_risk_states), e.g. (3, 3, 3).
        metadata = {
            "evidence_nodes": self._evidence_nodes,
            "query_nodes": self._query_nodes,
        }
        arrays = dict(self._inference_table)
        arrays["__metadata__"] = np.array([json.dumps(metadata)])
        np.savez_compressed(path, **arrays)
        _log.info("Saved precomputed table to '%s'", path)

    def load_precomputed(self, path: str | Path) -> None:
        """Load a precomputed lookup table saved with :meth:`save_precomputed`.

        After loading, :meth:`infer` uses the table path (O(H×W) numpy
        indexing) without calling pgmpy.

        Parameters
        ----------
        path:
            Path to the ``.npz`` file written by :meth:`save_precomputed`.

        Raises
        ------
        FileNotFoundError
            If neither *path* nor *path* + ``.npz`` exists.
        ValueError
            If the table's node order, query nodes, or array shapes do not
            match the current BN configuration.
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

        # Validate node order matches current inputs
        current_order = list(self._inputs.keys())
        if node_order != current_order:
            raise ValueError(
                f"Node order mismatch: table has {node_order}, "
                f"current inputs are {current_order}.  "
                "Re-register inputs in the same order or re-run precompute()."
            )

        # Validate every query node exists in the BN
        for n in query_nodes:
            self._validate_node_exists(n)

        # Validate array shapes match current discretizations
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

        self._inference_table = {
            qnode: np.array(data[qnode], dtype=np.float32) for qnode in query_nodes
        }
        self._evidence_nodes = node_order
        self._query_nodes = query_nodes
        _log.info(
            "Loaded precomputed table from '%s': query=%s, evidence=%s",
            path, query_nodes, node_order,
        )

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
            candidate_grids: list[tuple[float, GridSpec, str]] = []  # (pixel_size, grid, node)

            for node, source in self._inputs.items():
                if source.requires_grid:
                    continue  # needs bbox first — skip for now
                data = source.fetch(grid=None)
                pre_fetched[node] = data
                if data.crs is None:
                    continue  # ConstantSource or similar — no spatial info
                grid_candidate = GridSpec.from_raster_data(data)
                pixel_size = abs(grid_candidate.transform.a)
                candidate_grids.append((pixel_size, grid_candidate, node))

            if not candidate_grids:
                raise ValueError(
                    "Could not determine a reference grid automatically.  "
                    "All registered sources require a grid bbox before they "
                    "can fetch data (requires_grid=True).  "
                    "Call bn.set_grid(crs, resolution, extent) explicitly."
                )

            # Pick the source with the finest (smallest) pixel size so that
            # high-resolution sources are never downsampled unnecessarily.
            _smallest_pixel, ref_grid, ref_node = min(candidate_grids, key=lambda t: t[0])
            _log.info("Auto-selected reference grid from '%s' (finest resolution)", ref_node)

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
                idx = discretize_array(arr, spec)

                if node in self._frozen_nodes:
                    # Cache discrete array; also cache the grid so the next call
                    # can skip the first-node fetch that derives the grid.
                    self._frozen_cache[node] = idx
                    if self._cached_ref_grid is None:
                        self._cached_ref_grid = ref_grid

            nodata_mask |= idx < 0
            evidence_state_grids[node] = idx
            evidence_state_names[node] = spec.labels

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
