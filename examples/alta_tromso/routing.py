"""Least-cost routes over a raster, for the Alta → Tromsø example.

Route planning is not part of geobn.  This module shows how a risk map from
geobn feeds a planner: every water pixel is a node of a graph, joined to its
eight neighbours, and scipy's Dijkstra finds the cheapest path.

The cost of a step is its length times the mean *cost factor* of the two
pixels it joins.  A cost factor of 1 everywhere gives the shortest route; a
factor of ``1 + k * risk`` makes the planner trade distance for safety.
Pixels whose factor is NaN or infinite (land, no data) have no edges, so no
route can cross them.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

# Neighbour offsets (row, col) and their length in pixels.  Only half of the
# eight directions are listed: each undirected edge is added in both
# directions below.
_STEPS = [(0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2.0)), (1, -1, np.sqrt(2.0))]


def plan_route(
    cost_factor: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    pixel_size: float,
) -> tuple[np.ndarray, float, float]:
    """Return the cheapest 8-connected path from *start* to *goal*.

    Parameters
    ----------
    cost_factor:
        (H, W) cost per metre travelled in each pixel, at least 1 where
        passable, NaN or inf where impassable.
    start, goal:
        (row, col) of the end points.  Both must be passable.
    pixel_size:
        Pixel edge length in metres.

    Returns
    -------
    path:
        (N, 2) array of (row, col), from *start* to *goal*.
    length_m:
        Geometric length of the path in metres.
    cost:
        Total cost of the path (metres weighted by the cost factor).
    """
    h, w = cost_factor.shape
    passable = np.isfinite(cost_factor)
    for name, (r, c) in (("start", start), ("goal", goal)):
        if not passable[r, c]:
            raise ValueError(f"{name} {(r, c)} is not a passable pixel")

    index = np.arange(h * w).reshape(h, w)
    rows, cols, weights = [], [], []
    for dr, dc, step in _STEPS:
        # The block of pixels that has a neighbour at (dr, dc) inside the grid
        r0, r1 = 0, h - dr
        c0, c1 = max(0, -dc), w - max(0, dc)
        a = (slice(r0, r1), slice(c0, c1))
        b = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
        ok = passable[a] & passable[b]
        if dr and dc:
            # A diagonal step may not cut the corner between two land pixels.
            ok &= passable[r0:r1, c0 + dc:c1 + dc] | passable[r0 + dr:r1 + dr, c0:c1]
        weight = step * pixel_size * 0.5 * (cost_factor[a] + cost_factor[b])
        rows.append(index[a][ok])
        cols.append(index[b][ok])
        weights.append(weight[ok])

    src = np.concatenate(rows + cols)
    dst = np.concatenate(cols + rows)
    graph = coo_matrix(
        (np.concatenate(weights + weights), (src, dst)), shape=(h * w, h * w)
    ).tocsr()

    s = int(index[start])
    g = int(index[goal])
    dist, pred = dijkstra(graph, directed=True, indices=s, return_predecessors=True)
    if not np.isfinite(dist[g]):
        raise ValueError("goal cannot be reached from start over passable pixels")

    node = g
    path = [node]
    while node != s:
        node = int(pred[node])
        path.append(node)
    path = np.array(np.unravel_index(path[::-1], (h, w))).T
    steps = np.hypot(*np.diff(path, axis=0).T)
    return path, float(steps.sum() * pixel_size), float(dist[g])


def snap_to_water(passable: np.ndarray, rc: tuple[int, int]) -> tuple[int, int]:
    """Return the passable pixel nearest to *rc* (itself if passable)."""
    if passable[rc]:
        return rc
    rr, cc = np.nonzero(passable)
    i = int(np.argmin((rr - rc[0]) ** 2 + (cc - rc[1]) ** 2))
    return int(rr[i]), int(cc[i])
