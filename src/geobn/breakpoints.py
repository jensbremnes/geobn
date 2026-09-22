"""Compute discretization breakpoints from data.

The functions here turn an array of values into the breakpoint list
:meth:`~geobn.GeoBayesianNetwork.set_discretization` expects::

    arr = bn.fetch_raw(geobn.RasterSource("slope.tif"))
    bn.set_discretization("slope_angle", geobn.breakpoints.quantile(arr, 4))

Both schemes return ``n + 1`` ascending edges: ``n - 1`` interior bin edges
between the outer pair that defines the valid range.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = ["equal_interval", "quantile"]


def quantile(
    values: np.ndarray | Sequence[float],
    n: int,
    *,
    bounds: tuple[float | None, float | None] | None = None,
) -> list[float]:
    """Breakpoints that put roughly the same number of values in each bin.

    The interior edges are the n-quantiles of the data, so no state is left
    empty.  On skewed data the bins have very different widths, which is the
    price of even occupancy.

    Parameters
    ----------
    values:
        Array of any shape.  It is flattened, and NaN and infinite values are
        ignored, so a raster's NoData pixels need no special handling.
    n:
        Number of bins, i.e. the number of BN states for the node.  The
        returned list has ``n + 1`` entries.
    bounds:
        ``(lo, hi)`` for the outer breakpoints, which define the valid range
        ``[lo, hi]`` (see
        :meth:`~geobn.GeoBayesianNetwork.set_discretization`).  Either side may
        be ``None`` to take that side from the data, and the default ``None``
        takes both.  Pass a node's physical range — ``bounds=(0, 90)`` for a
        slope in degrees — when later data may go beyond the values at hand.
        ``(-inf, inf)`` leaves the range unbounded, so nothing is ever out of
        range.

    Returns
    -------
    list of float
        ``n + 1`` strictly increasing breakpoints.

    Raises
    ------
    ValueError
        If *n* is not a positive integer, *values* holds no finite value,
        *bounds* is malformed, an interior edge falls outside *bounds*, or the
        data is too concentrated to form *n* bins.

    Example
    -------
    >>> quantile([0, 1, 2, 3, 4, 5, 6, 7], 4)
    [0.0, 1.75, 3.5, 5.25, 7.0]
    """
    n = _check_n(n)
    lo_bound, hi_bound = _check_bounds(bounds)
    finite = _finite_values(values)

    edges = [float(e) for e in np.percentile(finite, np.linspace(0.0, 100.0, n + 1))]
    if lo_bound is not None:
        edges[0] = lo_bound
    if hi_bound is not None:
        edges[-1] = hi_bound

    _validate_edges(edges, finite, n, "quantile")
    return edges


def equal_interval(
    values: np.ndarray | Sequence[float],
    n: int,
    *,
    bounds: tuple[float | None, float | None] | None = None,
) -> list[float]:
    """Breakpoints that split the range into bins of equal width.

    The edges are round and easy to read, and they describe the range rather
    than the distribution: on skewed data most values can land in one bin while
    the others stay nearly empty.

    Parameters
    ----------
    values:
        Array of any shape.  It is flattened, and NaN and infinite values are
        ignored, so a raster's NoData pixels need no special handling.
    n:
        Number of bins, i.e. the number of BN states for the node.  The
        returned list has ``n + 1`` entries.
    bounds:
        ``(lo, hi)`` for the outer breakpoints, which define both the span that
        is divided and the valid range ``[lo, hi]``.  Either side may be
        ``None`` to take that side from the data, and the default ``None``
        takes both.  Both bounds must be finite, because bins of equal width
        over an infinite span are undefined; :func:`quantile` accepts infinite
        bounds.

    Returns
    -------
    list of float
        ``n + 1`` strictly increasing, evenly spaced breakpoints.

    Raises
    ------
    ValueError
        If *n* is not a positive integer, *values* holds no finite value,
        *bounds* is malformed or infinite, or the span is too narrow to form
        *n* bins.

    Example
    -------
    >>> equal_interval([0, 1, 2, 3, 4, 5, 6, 7], 4)
    [0.0, 1.75, 3.5, 5.25, 7.0]
    """
    n = _check_n(n)
    lo_bound, hi_bound = _check_bounds(bounds)
    finite = _finite_values(values)

    lo = float(finite.min()) if lo_bound is None else lo_bound
    hi = float(finite.max()) if hi_bound is None else hi_bound
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(
            f"equal_interval needs finite bounds, got ({lo:g}, {hi:g}): bins of "
            "equal width over an infinite span are undefined.  Use quantile() "
            "for unbounded outer breakpoints."
        )

    edges = [float(e) for e in np.linspace(lo, hi, n + 1)]
    _validate_edges(edges, finite, n, "equal-interval")
    return edges


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_n(n: int) -> int:
    """Validate the bin count."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"n must be an integer number of bins; got {n!r}")
    if n < 1:
        raise ValueError(f"n must be at least 1 bin; got {n}")
    return int(n)


def _check_bounds(
    value: tuple[float | None, float | None] | None,
) -> tuple[float | None, float | None]:
    """Validate a ``bounds`` argument and normalise it to floats or None."""
    if value is None:
        return None, None
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            "bounds must be a (lo, hi) tuple, either bound optionally None to "
            f"take that side from the data; got {value!r}"
        )

    out: list[float | None] = []
    for bound in value:
        if bound is None:
            out.append(None)
            continue
        if isinstance(bound, bool) or not isinstance(bound, (int, float, np.number)):
            raise ValueError(f"bounds must be numbers or None; got {value!r}")
        bound = float(bound)
        if math.isnan(bound):
            raise ValueError(f"bounds must not be NaN; got {value!r}")
        out.append(bound)

    lo, hi = out
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError(f"bounds lower must be below the upper bound; got {value!r}")
    return lo, hi


def _finite_values(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Flatten *values* to a 1-D float array holding only its finite entries."""
    try:
        array = np.asarray(values, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"values must be an array of numbers; got {type(values).__name__} ({exc})"
        ) from exc

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(
            "No finite values to compute breakpoints from: the data is empty, "
            "all NaN, or all infinite."
        )
    return finite


def _validate_edges(
    edges: list[float], finite: np.ndarray, n: int, scheme: str
) -> None:
    """Check computed *edges* are usable as a discretization's breakpoints."""
    lo, hi = edges[0], edges[-1]
    for edge in edges[1:-1]:
        if edge < lo or edge > hi:
            raise ValueError(
                f"Interior breakpoint {edge:g} falls outside the bounds "
                f"({lo:g}, {hi:g}), which the data ({finite.min():g} to "
                f"{finite.max():g}) does not fit inside.  Widen bounds= to span "
                "the data, or drop it to use the data's own range."
            )

    for i in range(1, len(edges)):
        if edges[i] > edges[i - 1]:
            continue
        repeated = edges[i]
        share = float(np.mean(finite == repeated))
        raise ValueError(
            f"Cannot form {n} {scheme} bins: {share:.0%} of the values are "
            f"{repeated:g}, so breakpoints {i - 1} and {i} are both "
            f"{repeated:g}.  Use fewer bins, or set the breakpoints explicitly."
        )
