from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .._types import RasterData

if TYPE_CHECKING:
    from ..grid import GridSpec

_log = logging.getLogger(__name__)


class DataSource(ABC):
    """Base class for all geographic data sources.

    Every source returns a plain RasterData tuple (array, crs, transform)
    — no rasterio objects are ever exposed outside the source module.

    Subclasses implement :meth:`_fetch`; :meth:`fetch` wraps it, reads and
    writes the disk cache, and applies ``valid_range``.  A source becomes
    cacheable by overriding :meth:`_cache_key`.

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
    cache_dir:
        Optional directory for caching fetched data on disk.  Accepted only
        by sources that can be cached.
    cache_ttl:
        Maximum age of a cache entry, as a :class:`~datetime.timedelta` or a
        number of seconds.  An older entry is refetched.  The default
        ``None`` never expires, which suits data that does not change, such
        as terrain.  If the refetch fails and an expired entry exists, that
        entry is returned with a :class:`UserWarning` giving its age.
    """

    # Subclasses that require a grid bbox before they can fetch (e.g. WCS,
    # Open-Meteo) override this to True.  Self-contained sources (Raster,
    # Array, URL, Constant) keep the default False.
    requires_grid: bool = False

    def __init__(
        self,
        valid_range: tuple[float | None, float | None] | None = None,
        cache_dir: str | Path | None = None,
        cache_ttl: timedelta | float | None = None,
    ) -> None:
        self._valid_range = _check_valid_range(valid_range)
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self._cache_ttl = _check_cache_ttl(cache_ttl)

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        """Fetch data aligned to *grid* if provided.

        Serves the disk cache when one is configured and the entry is within
        ``cache_ttl``.  Values outside ``valid_range`` are returned as NaN.

        Parameters
        ----------
        grid:
            Reference grid context.  Required by sources that need to know the
            spatial domain before querying (e.g. OpenMeteoSource).  Ignored by
            sources that are self-contained (Array, Raster, URL, Constant).
        """
        return _apply_valid_range(self._fetch_cached(grid), self._valid_range)

    @abstractmethod
    def _fetch(self, grid: GridSpec | None = None) -> RasterData:
        """Return the source's data as RasterData.  Implemented by subclasses."""

    def _cache_key(self, grid: GridSpec | None = None) -> dict | None:
        """Return a JSON-serialisable dict identifying this request.

        Returning ``None`` — the default — means the source is not cached.
        The key must not include ``cache_ttl``: freshness is a property of the
        entry, not of what was requested.
        """
        return None

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    def _fetch_cached(self, grid: GridSpec | None = None) -> RasterData:
        """Call :meth:`_fetch`, reading and writing the disk cache around it."""
        key = self._cache_key(grid) if self._cache_dir is not None else None
        if key is None:
            return self._fetch(grid)

        from ._cache import (  # noqa: PLC0415
            _cache_age,
            _load_cached,
            _make_cache_path,
            _save_cached,
        )

        cache_path = _make_cache_path(self._cache_dir, key)
        cached = _load_cached(cache_path, ttl=self._cache_ttl)
        if cached is not None:
            return cached

        try:
            result = self._fetch(grid)
        except Exception:
            # An expired entry beats no data at all, but say so out loud.
            stale = _load_cached(cache_path)
            if stale is None:
                raise
            age = _cache_age(cache_path)
            warnings.warn(
                f"Could not refresh {type(self).__name__}; using cached data "
                f"{_format_age(age)} old (cache_ttl={_format_age(self._cache_ttl)}).",
                UserWarning,
                stacklevel=3,
            )
            return stale

        _save_cached(cache_path, result)
        return result


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


def _check_cache_ttl(value: timedelta | float | None) -> float | None:
    """Validate a ``cache_ttl`` argument and normalise it to seconds."""
    if value is None:
        return None
    if isinstance(value, timedelta):
        seconds = value.total_seconds()
    elif isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(
            "cache_ttl must be a timedelta or a number of seconds; "
            f"got {value!r}"
        )
    else:
        seconds = float(value)
    if math.isnan(seconds):
        raise ValueError(f"cache_ttl must not be NaN; got {value!r}")
    if seconds < 0:
        raise ValueError(f"cache_ttl must not be negative; got {value!r}")
    return seconds


def _format_age(seconds: float | None) -> str:
    """Render an age in seconds for a warning message."""
    if seconds is None:
        return "an unknown time"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172_800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86_400:.1f} days"


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
