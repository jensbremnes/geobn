"""Batched pixel-wise Bayesian network inference.

Strategy
--------
Rather than running pgmpy once per pixel (potentially millions of times),
we find all *unique combinations* of discretised input states and look up
the posterior once per combination.  Two strategies cover the spectrum:

1. **Per-combo loop** — one VariableElimination query per unique evidence
   combination (all query nodes share a single elimination pass).  Cheapest
   when only a handful of combinations occur.
2. **Single joint query** — for larger combination counts, the *full*
   conditional table P(query | e_1 ... e_k) is obtained from one VE query
   of the joint P(query, e_1, ..., e_k) normalised along the query axis.
   One pgmpy call replaces up to prod(n_states) calls; results are then
   mapped to pixels by numpy fancy indexing.  Used whenever the table fits
   within ``_MAX_TABLE_CELLS``.

Data flow
---------
evidence_state_grids  dict[node, (H, W) int16]   state index per pixel
                  (-1 = NoData)
nodata_mask       (H, W) bool                True where any input is NaN

Returns
-------
dict[node, (H, W, n_states) float32]         probability per pixel per state
"""
from __future__ import annotations

import logging

import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

_log = logging.getLogger(__name__)

# Above this many unique evidence combinations, run_inference switches from
# the per-combo VE loop to a single joint query (when the table fits in
# memory).  Below it, the loop is cheaper than materialising the full table.
_COMBO_LOOP_THRESHOLD = 200

# Upper bound on conditional-table size (total float32 cells across all query
# nodes, ~80 MB at the default).  Beyond this the joint-query strategy is
# skipped and the per-combo loop is used instead.
_MAX_TABLE_CELLS = 20_000_000


def build_conditional_table(
    model: DiscreteBayesianNetwork,
    evidence_nodes: list[str],
    query_nodes: list[str],
    ve: VariableElimination | None = None,
    max_table_cells: int = _MAX_TABLE_CELLS,
) -> dict[str, np.ndarray] | None:
    """Compute P(query | evidence) for *all* evidence combinations at once.

    Instead of one VE query per evidence combination (prod(n_states) calls),
    the joint P(query, e_1, ..., e_k) is computed with a *single* VE query
    per query node and normalised along the query axis.  The result is a
    lookup table identical in shape to the one built by
    :meth:`~geobn.GeoBayesianNetwork.precompute`.

    Parameters
    ----------
    model:
        A fitted pgmpy DiscreteBayesianNetwork.
    evidence_nodes:
        Evidence node names; their order defines the table axes.
    query_nodes:
        Nodes whose conditional distributions are tabulated.
    ve:
        Optional pre-built VariableElimination engine.
    max_table_cells:
        If the total number of table cells (summed over query nodes) exceeds
        this bound, *None* is returned and the caller should fall back to
        per-combination queries.

    Returns
    -------
    Mapping from query node to a float32 array of shape
    ``(n_states_0, ..., n_states_k, n_query_states)``, or *None* if the
    table would exceed *max_table_cells* (or a query node is itself an
    evidence node).  Evidence combinations with zero prior probability
    yield NaN rows.
    """
    if any(q in evidence_nodes for q in query_nodes):
        return None  # joint over duplicated variables is ill-defined; caller falls back

    n_states_evidence = [
        len(model.get_cpds(n).state_names[n]) for n in evidence_nodes
    ]
    prod_evidence = 1
    for k in n_states_evidence:
        prod_evidence *= k
    total_cells = sum(
        prod_evidence * len(model.get_cpds(q).state_names[q]) for q in query_nodes
    )
    if total_cells > max_table_cells:
        _log.info(
            "Conditional table would need %d cells (> %d) — falling back to "
            "per-combination queries", total_cells, max_table_cells,
        )
        return None

    if ve is None:
        ve = VariableElimination(model)

    tables: dict[str, np.ndarray] = {}
    for qnode in query_nodes:
        factor = ve.query([qnode, *evidence_nodes], show_progress=False)

        # pgmpy orders factor axes by its own variable bookkeeping, not by the
        # requested order — transpose to (evidence_nodes..., qnode).
        axis_order = [factor.variables.index(n) for n in (*evidence_nodes, qnode)]
        for var in factor.variables:
            expected = list(model.get_cpds(var).state_names[var])
            if list(factor.state_names[var]) != expected:
                raise RuntimeError(
                    f"pgmpy returned states {factor.state_names[var]} for "
                    f"'{var}', expected CPD order {expected}"
                )
        joint = np.transpose(factor.values, axis_order)

        # Normalise the joint into P(query | evidence).  Zero-probability
        # evidence combinations divide 0/0 → NaN, mirroring what a direct
        # query with impossible evidence would produce.
        with np.errstate(invalid="ignore", divide="ignore"):
            tables[qnode] = (
                joint / joint.sum(axis=-1, keepdims=True)
            ).astype(np.float32)

    return tables


def _unique_evidence_combos(
    state_matrix: np.ndarray,
    n_states_per_node: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Find unique rows of *state_matrix* (n_pixels, n_nodes) and the inverse map.

    Equivalent to ``np.unique(state_matrix, axis=0, return_inverse=True)`` but
    packs each row into a single int64 key first — 1-D unique avoids the slow
    lexicographic row sort and is several times faster on large rasters.
    """
    packed_capacity = 1
    for k in n_states_per_node:
        packed_capacity *= k
    if packed_capacity > 2**62:
        # State space too large to pack into int64 — use the generic row unique.
        return np.unique(state_matrix, axis=0, return_inverse=True)

    keys = np.zeros(state_matrix.shape[0], dtype=np.int64)
    for i, k in enumerate(n_states_per_node):
        keys = keys * k + state_matrix[:, i]
    unique_keys, pixel_to_combo = np.unique(keys, return_inverse=True)

    # Decode packed keys back into one state index per node (reverse order).
    unique_combos = np.empty((len(unique_keys), len(n_states_per_node)), dtype=np.int32)
    remainder = unique_keys.copy()
    for i in range(len(n_states_per_node) - 1, -1, -1):
        unique_combos[:, i] = remainder % n_states_per_node[i]
        remainder //= n_states_per_node[i]
    return unique_combos, pixel_to_combo


def _query_per_combo(
    ve: VariableElimination,
    unique_combos: np.ndarray,
    node_list: list[str],
    evidence_state_names: dict[str, list[str]],
    query_nodes: list[str],
) -> dict[str, np.ndarray]:
    """Run one VE query per unique evidence combination.

    All query nodes are requested in a single ``joint=False`` call per combo,
    so the elimination pass is shared rather than repeated per query node.

    Returns a mapping from query node to a (n_unique, n_states) float32 array.
    """
    combo_probs: dict[str, list[np.ndarray]] = {q: [] for q in query_nodes}

    for combo in unique_combos:
        # combo holds one integer state index per evidence node; translate back
        # to the string state labels that pgmpy's query() expects.
        evidence_collection = {
            node_list[i]: evidence_state_names[node_list[i]][combo[i]]
            for i in range(len(node_list))
        }
        marginals = ve.query(
            query_nodes, evidence=evidence_collection, joint=False, show_progress=False
        )
        if not isinstance(marginals, dict):
            # Single query node: pgmpy may return the factor directly.
            marginals = {query_nodes[0]: marginals}
        for query_node in query_nodes:
            combo_probs[query_node].append(
                marginals[query_node].values.astype(np.float32)
            )

    return {q: np.stack(combo_probs[q], axis=0) for q in query_nodes}


def run_inference(
    model: DiscreteBayesianNetwork,
    evidence_state_grids: dict[str, np.ndarray],
    evidence_state_names: dict[str, list[str]],
    query_nodes: list[str],
    query_state_names: dict[str, list[str]],
    nodata_mask: np.ndarray,
    ve: VariableElimination | None = None,
) -> dict[str, np.ndarray]:
    """Run batched pixel-wise inference.

    Parameters
    ----------
    model:
        A fitted pgmpy BayesianNetwork.
    evidence_state_grids:
        Mapping from evidence node name to (H, W) int16 array of state indices.
    evidence_state_names:
        Mapping from evidence node name to its ordered list of state labels.
    query_nodes:
        Nodes whose posterior distributions are requested.
    query_state_names:
        Mapping from query node name to its ordered list of state labels.
    nodata_mask:
        (H, W) boolean array; True where any input pixel is NoData.
    ve:
        Pre-built :class:`pgmpy.inference.VariableElimination` engine.  If
        *None* (default) a new one is created from *model*.  Pass a cached
        instance to avoid recreating it on every call when the model does not
        change.

    Returns
    -------
    Mapping from query node name to a (H, W, n_states) float32 array.
    """
    H, W = next(iter(evidence_state_grids.values())).shape
    node_list = list(evidence_state_grids.keys())
    n_states_per_node = [len(evidence_state_names[n]) for n in node_list]

    valid = ~nodata_mask  # (H, W)
    n_valid = int(valid.sum())

    # Pre-allocate output arrays filled with NaN
    output: dict[str, np.ndarray] = {}
    for query_node in query_nodes:
        n_states = len(query_state_names[query_node])
        output[query_node] = np.full((H, W, n_states), np.nan, dtype=np.float32)

    if n_valid == 0:
        return output

    # Matrix where each row is a valid pixel and each column is an evidence node.
    # The value in each cell is the state index of that node. Dim (n_valid, n_nodes).
    valid_pixel_state_matrix = np.column_stack(
        [evidence_state_grids[n][valid].astype(np.int32) for n in node_list]
    )

    # Find all distinct combinations of evidence states that appear across valid pixels.
    # If two pixels have identical combinations of evidence states, they appear as one row.
    #
    # unique_combos:  one row per distinct combination, e.g. [[0,1], [2,0], [2,2]]
    # pixel_to_combo: one entry per valid pixel — the row index in unique_combos that
    #                 pixel belongs to, e.g. [0, 0, 1, 2, 0, ...]
    unique_combos, pixel_to_combo = _unique_evidence_combos(
        valid_pixel_state_matrix, n_states_per_node
    )

    _log.info(
        "Inference: %d×%d grid, %d/%d valid pixels, %d unique evidence combination(s)",
        H, W, n_valid, H * W, len(unique_combos),
    )

    if ve is None:
        ve = VariableElimination(model)

    # For each query node, one probability distribution per unique evidence
    # combination, row-aligned with unique_combos: shape (n_unique, n_states).
    probs_per_combo: dict[str, np.ndarray] | None = None

    if len(unique_combos) > _COMBO_LOOP_THRESHOLD:
        # Many combinations: one joint VE query yields the full conditional
        # table; per-combo distributions are then read by fancy indexing.
        tables = build_conditional_table(model, node_list, query_nodes, ve=ve)
        if tables is not None:
            _log.info(
                "Using single-joint-query strategy (%d combos, 1 pgmpy query "
                "per query node)", len(unique_combos),
            )
            combo_index = tuple(unique_combos[:, i] for i in range(len(node_list)))
            probs_per_combo = {q: tables[q][combo_index] for q in query_nodes}

    if probs_per_combo is None:
        probs_per_combo = _query_per_combo(
            ve, unique_combos, node_list, evidence_state_names, query_nodes
        )

    # Map inference results back to the spatial grid.
    # For each query node, every valid pixel is assigned the probability distribution
    # of its evidence combination, then written into the correct position in the output grid.
    for query_node in query_nodes:

        # Use pixel_to_combo to give each valid pixel the distribution of its combo: row i = distribution for pixel i.
        valid_pixel_probs = probs_per_combo[query_node][pixel_to_combo]  # (n_valid, n_states)

        # Write the probabilities into the valid pixel slots of the output grid —
        # a 3D array (H, W, n_states) where each pixel holds one probability per state.
        # NaN pixels are left untouched.
        output[query_node][valid] = valid_pixel_probs

    return output


def run_inference_from_table(
    table: dict[str, np.ndarray],
    node_order: list[str],
    evidence_state_grids: dict[str, np.ndarray],
    nodata_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map pixel-wise discrete evidence to precomputed probabilities via fancy indexing.

    This is the zero-pgmpy fast path used after
    :meth:`~geobn.GeoBayesianNetwork.precompute`.  Probabilities are read from
    a lookup table using numpy advanced indexing — O(H×W) rather than running
    pgmpy per unique evidence combination.

    Parameters
    ----------
    table:
        Mapping from query node name to a numpy array of shape
        ``(n_states_0, n_states_1, ..., n_states_k, n_query_states)`` where
        the first *k* axes correspond to the *k* nodes in *node_order*.
    node_order:
        Evidence node names in the order matching the table axes.
    evidence_state_grids:
        Mapping from node name to ``(H, W)`` int array of state indices.
        Nodata pixels (index -1) are masked out via *nodata_mask*.
    nodata_mask:
        ``(H, W)`` boolean array; True where any input pixel is NoData.

    Returns
    -------
    Mapping from query node name to a ``(H, W, n_states)`` float32 array.
    NaN where *nodata_mask* is True.
    """
    H, W = nodata_mask.shape
    n_valid = int((~nodata_mask).sum())
    _log.info("Table lookup: %d×%d grid, %d valid pixels (fast path, no pgmpy)", H, W, n_valid)

    # One state-grid per evidence node, ordered to match the axes of the precomputed
    # table. Used as a combined index so numpy can read the right probabilities for
    # every pixel in one operation rather than looping over them.
    node_state_index_tuple = tuple(evidence_state_grids[n] for n in node_order)

    output: dict[str, np.ndarray] = {}
    for node, tbl in table.items():
        n_states = tbl.shape[-1]
        probs = np.asarray(tbl[node_state_index_tuple], dtype=np.float32)
        # broadcast_to handles the edge case where all indices happen to be scalars
        probs = np.broadcast_to(probs, (H, W, n_states)).copy()
        probs[nodata_mask] = np.nan
        output[node] = probs

    return output


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute per-pixel Shannon entropy (bits) from a probability array.

    Parameters
    ----------
    probs:
        (..., n_states) array of probabilities.

    Returns
    -------
    (...) array of entropy values.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # log2(0) is -inf, but by convention 0 * log2(0) = 0 (zero-probability
        # states contribute nothing to entropy).  np.where substitutes 0.0 for
        # those terms before the multiplication.
        log2_p = np.where(probs > 0, np.log2(probs), 0.0)
    return -np.sum(probs * log2_p, axis=-1)
