from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .._types import RasterData

if TYPE_CHECKING:
    from ..grid import GridSpec


class DataSource(ABC):
    """Base class for all geographic data sources.

    Every source returns a plain RasterData tuple (array, crs, transform)
    — no rasterio objects are ever exposed outside the source module.

    Subclasses implement :meth:`_fetch`; :meth:`fetch` wraps it and applies
    ``valid_range``.

    Parameters
    ----------
    valid_range:
        Optional ``(lo, hi)`` tuple.  Pixels below *lo* or above *hi* are
        replaced with NaN, so they are excluded from inference instead of
        being discretised as real values.  Both bounds are inclusive, and
        either may be ``None`` to leave that side unbounded.  Use this for
        data that encodes missing values as extreme numbers without
        declaring them as nodata, e.g. ``(-500, 9000)`` for elevation in
        metres.
    """

    # Subclasses that require a grid bbox before they can fetch (e.g. WCS,
    # Open-Meteo) override this to True.  Self-contained sources (Raster,
    # Array, URL, Constant) keep the default False.
    requires_grid: bool = False

    def __init__(
        self, valid_range: tuple[float | None, float | None] | None = None
    ) -> None:
        self._valid_range = _check_valid_range(valid_range)

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        """Fetch data aligned to *grid* if provided.

        Values outside ``valid_range`` are returned as NaN.

        Parameters
        ----------
        grid:
            Reference grid context.  Required by sources that need to know the
            spatial domain before querying (e.g. OpenMeteoSource).  Ignored by
            sources that are self-contained (Array, Raster, URL, Constant).
        """
        return _apply_valid_range(self._fetch(grid), self._valid_range)

    @abstractmethod
    def _fetch(self, grid: GridSpec | None = None) -> RasterData:
        """Return the source's data as RasterData.  Implemented by subclasses."""


def _check_valid_range(
    value: tuple[float | None, float | None] | None,
) -> tuple[float | None, float | None] | None:
    """Validate a ``valid_range`` argument and normalise it to floats."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            "valid_range must be a (lo, hi) tuple, either bound optionally "
            f"None for an unbounded side; got {value!r}"
        )

    bounds: list[float | None] = []
    for bound in value:
        if bound is None:
            bounds.append(None)
            continue
        if isinstance(bound, bool) or not isinstance(bound, (int, float, np.number)):
            raise ValueError(
                f"valid_range bounds must be numbers or None; got {value!r}"
            )
        bound = float(bound)
        if math.isnan(bound):
            raise ValueError(f"valid_range bounds must not be NaN; got {value!r}")
        bounds.append(bound)

    lo, hi = bounds
    if lo is None and hi is None:
        raise ValueError(
            "valid_range needs at least one bound; use None to disable masking "
            f"instead of {value!r}"
        )
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"valid_range lower bound must not exceed the upper; got {value!r}")
    return lo, hi


def _apply_valid_range(
    data: RasterData, valid_range: tuple[float | None, float | None] | None
) -> RasterData:
    """Return *data* with values outside *valid_range* replaced by NaN."""
    if valid_range is None:
        return data

    lo, hi = valid_range
    array = data.array
    mask = np.zeros(array.shape, dtype=bool)
    if lo is not None:
        mask |= array < lo
    if hi is not None:
        mask |= array > hi
    if not mask.any():
        return data

    # Copy rather than mask in place: the array may be owned by the caller
    # (ArraySource holds on to it across fetches).
    masked = array.astype(np.float32) if array.dtype.kind != "f" else array.copy()
    masked[mask] = np.nan
    return data._replace(array=masked)
