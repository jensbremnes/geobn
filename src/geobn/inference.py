"""Batched pixel-wise Bayesian network inference.

Strategy
--------
Rather than running pgmpy once per pixel (potentially millions of times),
we find all *unique combinations* of discretised input states and look up
the posterior once per combination.  Two strategies cover the spectrum:

1. **Per-combo loop** — VariableElimination queries per unique evidence
   combination.  Cheapest when only a handful of combinations occur.  Within
   a combination, all query nodes are requested in one call when their joint
   table is small (``_MAX_JOINT_QUERY_CELLS``); otherwise each query node is
   queried separately.  pgmpy builds the full joint over all query nodes even
   with ``joint=False``, so one call for many query nodes can be orders of
   magnitude slower than separate calls.
2. **Single joint query** — for larger combination counts, the *full*
   conditional table P(query | e_1 ... e_k) is obtained from one VE query
   of the joint P(query, e_1, ..., e_k) normalised along the query axis.
   One pgmpy call replaces up to prod(n_states) calls; results are then
   mapped to pixels by numpy fancy indexing.  Used whenever the table fits
   within ``_MAX_TABLE_CELLS``.

Impossible evidence
-------------------
When an evidence combination has probability zero, the posterior is
undefined and every query node gets NaN on every path.  pgmpy on its own
would return numbers for query nodes that are d-separated from the
contradiction (it prunes that evidence away), so the per-combo loop checks
P(evidence) explicitly.

Missing inputs
--------------
A pixel where some inputs are NoData is solved with the inputs it does have.
A query node is NaN there only if a missing input is d-connected to it given
the observed inputs; otherwise its posterior does not depend on the missing
inputs and is computed as usual.  Pixels are grouped by which inputs are
missing (usually only a few patterns), and each pattern is solved with its
observed inputs as evidence.

Data flow
---------
evidence_state_grids  dict[node, (H, W) int16]   state index per pixel
                  (-1 = NoData)

Returns
-------
dict[node, (H, W, n_states) float32]         probability per pixel per state
"""
from __future__ import annotations

import logging
from collections.abc import Callable

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

# Upper bound on the joint table over the query nodes (product of their state
# counts) for requesting them in one pgmpy call.  Above it, each query node is
# queried separately.  Measured crossover is ~1e5 cells; this leaves margin.
_MAX_JOINT_QUERY_CELLS = 20_000

# Maps a set of missing input nodes to the query nodes it makes NaN.
BlankedBy = Callable[[frozenset[str]], frozenset[str]]


def make_blanked_by(
    model: DiscreteBayesianNetwork,
    evidence_nodes: list[str],
    query_nodes: list[str],
) -> BlankedBy:
    """Return a function giving the query nodes a set of missing inputs blanks.

    A query node is blanked when any missing input is d-connected to it given
    the inputs that are observed.  Results are memoised per missing set.
    """
    evidence_nodes = list(evidence_nodes)
    query = frozenset(query_nodes)
    cache: dict[frozenset[str], frozenset[str]] = {}

    def blanked_by(missing: frozenset[str]) -> frozenset[str]:
        if not missing:
            return frozenset()
        if missing not in cache:
            observed = [n for n in evidence_nodes if n not in missing]
            trails = model.active_trail_nodes(sorted(missing), observed=observed)
            reachable = set().union(*trails.values())
            cache[missing] = query & reachable
        return cache[missing]

    return blanked_by


def _missing_patterns(missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique rows of the (n, k) bool matrix *missing* and the inverse map."""
    n, k = missing.shape
    if k > 62:
        patterns, inverse = np.unique(missing, axis=0, return_inverse=True)
        return patterns, inverse.reshape(-1)
    keys = np.zeros(n, dtype=np.int64)
    for i in range(k):
        keys = keys * 2 + missing[:, i]
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    shifts = np.arange(k - 1, -1, -1, dtype=np.int64)
    patterns = ((unique_keys[:, None] >> shifts) & 1).astype(bool)
    return patterns, inverse.reshape(-1)


def _state_matrix(
    evidence_state_grids: dict[str, np.ndarray], node_order: list[str]
) -> tuple[tuple[int, ...], np.ndarray]:
    """Leading shape of the grids and their (n, k) matrix of state indices."""
    shape = np.shape(evidence_state_grids[node_order[0]])
    matrix = np.column_stack(
        [np.asarray(evidence_state_grids[n]).reshape(-1).astype(np.int32) for n in node_order]
    )
    return shape, matrix


def blank_masks(
    evidence_state_grids: dict[str, np.ndarray],
    node_order: list[str],
    query_nodes: list[str],
    blanked_by: BlankedBy | None = None,
) -> dict[str, np.ndarray]:
    """Where each query node is NaN because an input it depends on is missing.

    Returns one bool array per query node, shaped like the state grids.
    Without *blanked_by*, any missing input blanks every query node.
    """
    shape, matrix = _state_matrix(evidence_state_grids, node_order)
    missing = matrix < 0
    masks = {q: np.zeros(matrix.shape[0], dtype=bool) for q in query_nodes}
    patterns, pixel_to_pattern = _missing_patterns(missing)
    for p, pattern in enumerate(patterns):
        if not pattern.any():
            continue
        missing_nodes = frozenset(n for n, m in zip(node_order, pattern) if m)
        blanked = set(query_nodes) if blanked_by is None else blanked_by(missing_nodes)
        in_pattern = pixel_to_pattern == p
        for q in query_nodes:
            if q in blanked:
                masks[q] |= in_pattern
    return {q: m.reshape(shape) for q, m in masks.items()}


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
        # The joint over a duplicated variable is ill-defined.  GeoBayesianNetwork
        # rejects this before it gets here; direct callers of this function fall
        # back to the per-combination loop, where pgmpy raises.
        return None

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


def _root_priors(model: DiscreteBayesianNetwork) -> dict[str, dict[str, float]]:
    """Prior P(state) for every root node, keyed by node and state name."""
    priors: dict[str, dict[str, float]] = {}
    for node in model.nodes():
        if not list(model.predecessors(node)):
            cpd = model.get_cpds(node)
            priors[node] = dict(zip(cpd.state_names[node], cpd.get_values()[:, 0]))
    return priors


def _evidence_is_impossible(
    ve: VariableElimination,
    evidence: dict[str, str],
    root_priors: dict[str, dict[str, float]],
) -> bool:
    """True if P(evidence) == 0.

    Root nodes are marginally independent, so their joint probability is the
    product of their priors.  Remaining (non-root) evidence nodes are checked
    with the chain rule P(e_i | e_1 ... e_{i-1}), one single-node query each.
    """
    observed: dict[str, str] = {}
    for node, state in evidence.items():
        if node in root_priors:
            if not root_priors[node][state] > 0:
                return True
            observed[node] = state
    for node, state in evidence.items():
        if node in root_priors:
            continue
        factor = ve.query([node], evidence=observed, show_progress=False)
        if not factor.get_value(**{node: state}) > 0:
            return True
        observed[node] = state
    return False


def _query_marginals(
    ve: VariableElimination,
    model: DiscreteBayesianNetwork,
    query_nodes: list[str],
    evidence: dict[str, str],
    root_priors: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    """Posterior marginal of each query node given one evidence combination.

    Returns float32 arrays in BN state order.  All-NaN when P(evidence) == 0.
    """
    unique_nodes = list(dict.fromkeys(query_nodes))

    if _evidence_is_impossible(ve, evidence, root_priors):
        return {
            q: np.full(model.get_cardinality(q), np.nan, dtype=np.float32)
            for q in unique_nodes
        }

    joint_cells = 1
    for q in unique_nodes:
        joint_cells *= model.get_cardinality(q)

    if joint_cells <= _MAX_JOINT_QUERY_CELLS:
        marginals = ve.query(
            unique_nodes, evidence=evidence, joint=False, show_progress=False
        )
        if not isinstance(marginals, dict):
            # Single query node: pgmpy may return the factor directly.
            marginals = {unique_nodes[0]: marginals}
    else:
        marginals = {
            q: ve.query([q], evidence=evidence, show_progress=False)
            for q in unique_nodes
        }
    return {q: marginals[q].values.astype(np.float32) for q in unique_nodes}


def _query_per_combo(
    ve: VariableElimination,
    model: DiscreteBayesianNetwork,
    unique_combos: np.ndarray,
    node_list: list[str],
    evidence_state_names: dict[str, list[str]],
    query_nodes: list[str],
) -> dict[str, np.ndarray]:
    """Run VE queries for each unique evidence combination.

    Returns a mapping from query node to a (n_unique, n_states) float32 array.
    """
    root_priors = _root_priors(model)
    combo_probs: dict[str, list[np.ndarray]] = {q: [] for q in query_nodes}

    for combo in unique_combos:
        # combo holds one integer state index per evidence node; translate back
        # to the string state labels that pgmpy's query() expects.
        evidence_collection = {
            node_list[i]: evidence_state_names[node_list[i]][combo[i]]
            for i in range(len(node_list))
        }
        marginals = _query_marginals(
            ve, model, query_nodes, evidence_collection, root_priors
        )
        for query_node in combo_probs:
            combo_probs[query_node].append(marginals[query_node])

    return {q: np.stack(combo_probs[q], axis=0) for q in combo_probs}


def run_inference(
    model: DiscreteBayesianNetwork,
    evidence_state_grids: dict[str, np.ndarray],
    evidence_state_names: dict[str, list[str]],
    query_nodes: list[str],
    query_state_names: dict[str, list[str]],
    ve: VariableElimination | None = None,
) -> dict[str, np.ndarray]:
    """Run batched pixel-wise inference.

    A pixel with NoData (index -1) in some inputs is solved with the inputs it
    has.  A query node is NaN there only when a missing input is d-connected
    to it given the observed inputs (see :func:`make_blanked_by`).

    Parameters
    ----------
    model:
        A fitted pgmpy BayesianNetwork.
    evidence_state_grids:
        Mapping from evidence node name to (H, W) int16 array of state indices,
        -1 for NoData.
    evidence_state_names:
        Mapping from evidence node name to its ordered list of state labels.
    query_nodes:
        Nodes whose posterior distributions are requested.
    query_state_names:
        Mapping from query node name to its ordered list of state labels.
    ve:
        Pre-built :class:`pgmpy.inference.VariableElimination` engine.  If
        *None* (default) a new one is created from *model*.  Pass a cached
        instance to avoid recreating it on every call when the model does not
        change.

    Returns
    -------
    Mapping from query node name to a (H, W, n_states) float32 array.
    """
    node_list = list(evidence_state_grids.keys())
    (H, W), state_matrix = _state_matrix(evidence_state_grids, node_list)
    query_nodes = list(dict.fromkeys(query_nodes))

    # Pre-allocate output arrays filled with NaN, and flat (pixel, state) views
    # of them to write into.
    output: dict[str, np.ndarray] = {}
    for query_node in query_nodes:
        n_states = len(query_state_names[query_node])
        output[query_node] = np.full((H, W, n_states), np.nan, dtype=np.float32)
    flat_output = {q: output[q].reshape(H * W, -1) for q in query_nodes}

    if ve is None:
        ve = VariableElimination(model)
    blanked_by = make_blanked_by(model, node_list, query_nodes)

    # Solve each group of pixels that miss the same inputs with the inputs
    # they have, for the query nodes those missing inputs cannot affect.
    patterns, pixel_to_pattern = _missing_patterns(state_matrix < 0)
    for p, pattern in enumerate(patterns):
        missing = frozenset(n for n, m in zip(node_list, pattern) if m)
        wanted = [q for q in query_nodes if q not in blanked_by(missing)]
        if not wanted:
            continue
        pixels = np.flatnonzero(pixel_to_pattern == p)
        observed_cols = np.flatnonzero(~pattern)
        probs = _posteriors(
            model,
            ve,
            state_matrix[np.ix_(pixels, observed_cols)],
            [node_list[i] for i in observed_cols],
            evidence_state_names,
            wanted,
            n_grid_pixels=H * W,
        )
        for q in wanted:
            flat_output[q][pixels] = probs[q]

    return output


def _posteriors(
    model: DiscreteBayesianNetwork,
    ve: VariableElimination,
    state_matrix: np.ndarray,
    node_list: list[str],
    evidence_state_names: dict[str, list[str]],
    query_nodes: list[str],
    n_grid_pixels: int,
) -> dict[str, np.ndarray]:
    """Posteriors for pixels that observe every node in *node_list*.

    *state_matrix* is (n_pixels, n_nodes).  Returns a (n_pixels, n_states)
    float32 array per query node.
    """
    n_states_per_node = [len(evidence_state_names[n]) for n in node_list]

    # Find all distinct combinations of evidence states that appear across the pixels.
    # If two pixels have identical combinations of evidence states, they appear as one row.
    #
    # unique_combos:  one row per distinct combination, e.g. [[0,1], [2,0], [2,2]]
    # pixel_to_combo: one entry per pixel — the row index in unique_combos that
    #                 pixel belongs to, e.g. [0, 0, 1, 2, 0, ...]
    unique_combos, pixel_to_combo = _unique_evidence_combos(
        state_matrix, n_states_per_node
    )

    _log.info(
        "Inference: %d/%d pixels observing %d input(s), %d unique evidence combination(s)",
        len(state_matrix), n_grid_pixels, len(node_list), len(unique_combos),
    )

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
            probs_per_combo = {
                # With no evidence nodes the table is one distribution; keep the combo axis.
                q: tables[q][combo_index] if combo_index else tables[q][np.newaxis]
                for q in query_nodes
            }

    if probs_per_combo is None:
        probs_per_combo = _query_per_combo(
            ve, model, unique_combos, node_list, evidence_state_names, query_nodes
        )

    # Every pixel is assigned the probability distribution of its evidence
    # combination: row i = distribution for pixel i, shape (n_pixels, n_states).
    return {q: probs_per_combo[q][pixel_to_combo] for q in query_nodes}


def run_inference_from_table(
    table: dict[str, np.ndarray],
    node_order: list[str],
    evidence_state_grids: dict[str, np.ndarray],
    blanked_by: BlankedBy | None = None,
) -> dict[str, np.ndarray]:
    """Map pixel-wise discrete evidence to precomputed probabilities via fancy indexing.

    This is the zero-pgmpy fast path used after
    :meth:`~geobn.GeoBayesianNetwork.precompute`.  Probabilities are read from
    a lookup table using numpy advanced indexing — O(H×W) rather than running
    pgmpy per unique evidence combination.

    For a pixel with NoData in some inputs, a query node that *blanked_by*
    does not list is independent of those inputs given the observed ones, so
    every table row over the missing inputs' states holds the same posterior
    (or NaN where that state combination is impossible).  The first row
    without NaN is used; if there is none, the observed evidence itself has
    probability zero and the result is NaN.

    Parameters
    ----------
    table:
        Mapping from query node name to a numpy array of shape
        ``(n_states_0, n_states_1, ..., n_states_k, n_query_states)`` where
        the first *k* axes correspond to the *k* nodes in *node_order*.
    node_order:
        Evidence node names in the order matching the table axes.
    evidence_state_grids:
        Mapping from node name to an int array of state indices, -1 for NoData.
        All arrays have the same shape, e.g. ``(H, W)`` or ``(K,)``.
    blanked_by:
        Function from a set of missing input nodes to the query nodes it makes
        NaN, as returned by :func:`make_blanked_by`.  Without it, any missing
        input makes every query node NaN.

    Returns
    -------
    Mapping from query node name to a float32 array of the grids' shape plus
    a trailing ``n_states`` axis.
    """
    shape, state_matrix = _state_matrix(evidence_state_grids, node_order)
    missing = state_matrix < 0
    complete = ~missing.any(axis=1)
    _log.info(
        "Table lookup: %d pixels, %d with every input (fast path, no pgmpy)",
        len(state_matrix), int(complete.sum()),
    )

    # One state column per evidence node, ordered to match the axes of the precomputed
    # table. Used as a combined index so numpy can read the right probabilities for
    # every complete pixel in one operation rather than looping over them.
    complete_index = tuple(state_matrix[complete, i] for i in range(len(node_order)))

    flat: dict[str, np.ndarray] = {}
    for node, tbl in table.items():
        probs = np.full((len(state_matrix), tbl.shape[-1]), np.nan, dtype=np.float32)
        probs[complete] = tbl[complete_index]
        flat[node] = probs

    if blanked_by is not None and not complete.all():
        _fill_partial_from_table(table, node_order, state_matrix, blanked_by, flat)

    return {q: p.reshape(*shape, p.shape[-1]) for q, p in flat.items()}


def _fill_partial_from_table(
    table: dict[str, np.ndarray],
    node_order: list[str],
    state_matrix: np.ndarray,
    blanked_by: BlankedBy,
    flat: dict[str, np.ndarray],
) -> None:
    """Fill pixels with some NoData inputs for the query nodes they do not blank."""
    partial = np.flatnonzero((state_matrix < 0).any(axis=1))
    patterns, pixel_to_pattern = _missing_patterns(state_matrix[partial] < 0)
    k = len(node_order)
    for p, pattern in enumerate(patterns):
        missing = frozenset(n for n, m in zip(node_order, pattern) if m)
        wanted = [q for q in table if q not in blanked_by(missing)]
        if not wanted:
            continue
        pixels = partial[pixel_to_pattern == p]
        observed_cols = np.flatnonzero(~pattern)
        missing_cols = np.flatnonzero(pattern)
        n_states = [table[wanted[0]].shape[i] for i in observed_cols]
        combos, pixel_to_combo = _unique_evidence_combos(
            state_matrix[np.ix_(pixels, observed_cols)], n_states
        )
        combo_index = tuple(combos[:, j] for j in range(len(observed_cols)))
        rows = np.arange(len(combos))
        for q in wanted:
            # Observed axes first, then the missing ones, then the query states.
            arranged = np.moveaxis(table[q], [*observed_cols, *missing_cols], list(range(k)))
            block = arranged[combo_index] if combo_index else arranged[np.newaxis]
            block = block.reshape(len(combos), -1, block.shape[-1])
            finite = np.isfinite(block).all(axis=-1)
            picked = block[rows, finite.argmax(axis=1)].astype(np.float32)
            picked[~finite.any(axis=1)] = np.nan
            flat[q][pixels] = picked[pixel_to_combo]


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
