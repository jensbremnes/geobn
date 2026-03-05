"""Batched pixel-wise Bayesian network inference.

Strategy
--------
Rather than running pgmpy once per pixel (potentially millions of times),
we find all *unique combinations* of discretised input states and run
inference exactly once per combination.  For typical BNs with a small
number of discrete states per node the number of unique combinations is
tiny even for very large rasters.

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

import numpy as np
from pgmpy.inference import VariableElimination


def run_inference(
    model,
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

    valid = ~nodata_mask  # (H, W)
    n_valid = int(valid.sum())

    # Pre-allocate output arrays filled with NaN
    output: dict[str, np.ndarray] = {}
    for node in query_nodes:
        n_states = len(query_state_names[node])
        output[node] = np.full((H, W, n_states), np.nan, dtype=np.float32)

    if n_valid == 0:
        return output

    # For each evidence node, extract the state indices of valid pixels only and
    # cast to int32 (required for np.unique to handle all nodes with a uniform dtype).
    valid_indices = [evidence_state_grids[n][valid].astype(np.int32) for n in node_list]
    valid_stack = np.column_stack(valid_indices)  # (n_valid, n_nodes)

    # Find unique evidence combinations
    unique_combos, inverse = np.unique(valid_stack, axis=0, return_inverse=True)
    # unique_combos: (n_unique, n_nodes)
    # inverse:       (n_valid,)  maps each valid pixel → unique combo index

    if ve is None:
        ve = VariableElimination(model)

    # Results per unique combo: dict[node] → list of probability arrays
    combo_probs: dict[str, list[np.ndarray]] = {node: [] for node in query_nodes}

    for combo in unique_combos:
        # combo holds one integer state index per evidence node; translate back
        # to the string state labels that pgmpy's query() expects.
        evidence = {
            node_list[i]: evidence_state_names[node_list[i]][combo[i]]
            for i in range(len(node_list))
        }
        for node in query_nodes:
            factor = ve.query([node], evidence=evidence, show_progress=False)
            combo_probs[node].append(factor.values.astype(np.float32))

    # Map results back to spatial arrays
    for node in query_nodes:
        probs_per_combo = np.stack(combo_probs[node], axis=0)  # (n_unique, n_states)
        flat_probs = probs_per_combo[inverse]                   # (n_valid, n_states)
        output[node][valid] = flat_probs

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

    # Build index tuple: one (H, W) int array per evidence axis.
    # Advanced indexing on a table of shape (*n_states_per_node, n_query_states)
    # with k arrays of shape (H, W) produces output of shape (H, W, n_query_states).
    idx = tuple(evidence_state_grids[n] for n in node_order)

    output: dict[str, np.ndarray] = {}
    for node, tbl in table.items():
        n_states = tbl.shape[-1]
        probs = np.asarray(tbl[idx], dtype=np.float32)
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
