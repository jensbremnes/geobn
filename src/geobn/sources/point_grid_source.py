"""Generic point-grid sampling source."""
from __future__ import annotations

import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Callable

import numpy as np
from affine import Affine

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource

_log = logging.getLogger(__name__)


class PointGridSource(DataSource):
    """Sample a user-supplied callable over an N×N lat/lon grid.

    Builds a regular grid of *sample_points* × *sample_points* WGS84 points
    covering the inference bounding box, calls ``fn(lat, lon)`` at each point,
    and assembles the results into a coarse EPSG:4326 raster.
    ``align_to_grid()`` then bilinearly resamples this raster to the reference
    grid resolution.

    This is the generic primitive for any point-queryable data source (weather
    APIs, elevation services, custom models).  Pass a lambda or a regular
    function as ``fn``; return ``float("nan")`` or ``None`` for missing values.

    Parameters
    ----------
    fn:
        Callable with signature ``(lat: float, lon: float) -> float | None``.
        Return ``float("nan")`` or ``None`` for positions with no data.
    sample_points:
        Number of sample points per axis.  Total API calls = ``sample_points²``.
        Default is 5 (25 calls).  Use 1 for a single-point broadcast (equivalent
        to ``ConstantSource`` but fetched dynamically).
    delay:
        Seconds to sleep between successive calls.  Default 0.05 s — enough to
        be polite to most free REST APIs without slowing batch runs noticeably.
    valid_range:
        Optional ``(lo, hi)`` tuple.  Either bound may be ``None``.  A sample
        outside the range becomes NaN, for APIs that report missing data as a
        magic number rather than null.
    name:
        Short identifier for this source, e.g. ``"wave_height"``.  It is what
        the disk cache is keyed on, together with the bounding box and
        ``sample_points``, because a callable cannot identify itself — two
        sources built from the same factory are indistinguishable otherwise.
        Required when ``cache_dir`` is set.
    cache_dir:
        Optional path to a directory for caching the sampled lattice on disk.
        On a cache hit no calls to ``fn`` are made at all.
    cache_ttl:
        Maximum age of a cache entry, as a :class:`~datetime.timedelta` or a
        number of seconds.  An older entry is resampled.  Forecasts are the
        usual case for this, e.g. ``cache_ttl=timedelta(hours=6)``.  The
        default ``None`` never expires.
    """

    requires_grid = True

    def __init__(
        self,
        fn: Callable[[float, float], float | None],
        sample_points: int = 5,
        delay: float = 0.05,
        valid_range: tuple[float | None, float | None] | None = None,
        name: str | None = None,
        cache_dir: str | Path | None = None,
        cache_ttl: timedelta | float | None = None,
    ) -> None:
        super().__init__(
            valid_range=valid_range, cache_dir=cache_dir, cache_ttl=cache_ttl
        )
        if cache_dir is not None and not name:
            raise ValueError(
                "PointGridSource needs a name to cache on disk, e.g. "
                "name='wave_height'.  A callable cannot identify itself, so "
                "without one two sources would share a cache entry."
            )
        self._fn = fn
        self._sample_points = max(1, sample_points)
        self._delay = delay
        self._name = name

    def _cache_key(self, grid: GridSpec | None = None) -> dict | None:
        if grid is None:
            return None
        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        return {
            "name": self._name,
            "n": self._sample_points,
            "lon_min": round(lon_min, 8), "lat_min": round(lat_min, 8),
            "lon_max": round(lon_max, 8), "lat_max": round(lat_max, 8),
        }

    def _fetch(self, grid: GridSpec | None = None) -> RasterData:
        if grid is None:
            raise ValueError(
                "PointGridSource requires a grid context to determine the spatial "
                "domain.  This is provided automatically by GeoBayesianNetwork.infer()."
            )

        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        n = self._sample_points

        _log.info("PointGridSource: sampling %d×%d grid (%d calls)", n, n, n * n)

        # Build N×N meshgrid in north→south row order
        lats = np.linspace(lat_max, lat_min, n)
        lons = np.linspace(lon_min, lon_max, n)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        values = np.full((n, n), np.nan, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                raw = self._fn(float(lat_grid[i, j]), float(lon_grid[i, j]))
                if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                    values[i, j] = np.nan
                else:
                    values[i, j] = float(raw)
                if n > 1:
                    time.sleep(self._delay)

        valid = values[~np.isnan(values)]
        if valid.size > 0:
            _log.info("PointGridSource: done — values range %.2f–%.2f", valid.min(), valid.max())
        else:
            _log.info("PointGridSource: done — all values are NaN")

        if n == 1:
            # Single-point result — behaves like ConstantSource (no CRS)
            return RasterData(array=values, crs=None, transform=None)

        # Half-pixel outward shift so each sample sits at its cell centre
        pixel_h = (lat_max - lat_min) / (n - 1)
        pixel_w = (lon_max - lon_min) / (n - 1)
        transform = Affine(pixel_w, 0, lon_min - pixel_w / 2, 0, -pixel_h, lat_max + pixel_h / 2)
        return RasterData(array=values, crs="EPSG:4326", transform=transform)
