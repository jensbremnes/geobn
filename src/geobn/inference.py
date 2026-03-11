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

import logging

import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

_log = logging.getLogger(__name__)


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
    # For example, if slope and rainfall each have 3 states, there are at most 9 unique
    # combinations — and we only need to run inference once per combination, not once per pixel.
    # If two pixels have identical combinations of evidence states, they appear as one row.
    #
    # unique_combos:  one row per distinct combination, e.g. [[0,1], [2,0], [2,2]]
    # pixel_to_combo: one entry per valid pixel — the row index in unique_combos that
    #                 pixel belongs to, e.g. [0, 0, 1, 2, 0, ...]
    unique_combos, pixel_to_combo = np.unique(valid_pixel_state_matrix, axis=0, return_inverse=True)

    _log.info(
        "Inference: %d×%d grid, %d/%d valid pixels, %d unique evidence combination(s)",
        H, W, n_valid, H * W, len(unique_combos),
    )

    if ve is None:
        ve = VariableElimination(model)

    # For each query node, stores the inferred probability distribution for every
    # unique evidence combination. Distributions are appended in the same order as
    # unique_combos, so position 0 here always corresponds to row 0 there.
    combo_probs: dict[str, list[np.ndarray]] = {query_node: [] for query_node in query_nodes}

    for combo in unique_combos:
        # combo holds one integer state index per evidence node; translate back
        # to the string state labels that pgmpy's query() expects.
        # Each integer in combo is used as a position to look up the label in
        # evidence_state_names, e.g. index 2 → "steep".
        evidence_collection = {
            node_list[i]: evidence_state_names[node_list[i]][combo[i]]
            for i in range(len(node_list))
        }
        for query_node in query_nodes:
            # query_result is a pgmpy object wrapping the posterior distribution;
            # .values gives a 1D array of probabilities, one per state.
            query_result = ve.query([query_node], evidence=evidence_collection, show_progress=False)
            combo_probs[query_node].append(query_result.values.astype(np.float32))

    # Map inference results back to the spatial grid.
    # For each query node, every valid pixel is assigned the probability distribution
    # of its evidence combination, then written into the correct position in the output grid.
    for query_node in query_nodes:

        # Stack per-combo results into a 2D lookup: row i = distribution for combo i.
        probs_per_combo = np.stack(combo_probs[query_node], axis=0)  # (n_unique, n_states)

        # Use pixel_to_combo to give each valid pixel the distribution of its combo: row i = distribution for pixel i.
        valid_pixel_probs = probs_per_combo[pixel_to_combo]          # (n_valid, n_states)
        
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
