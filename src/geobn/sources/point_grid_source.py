"""Generic point-grid sampling source."""
from __future__ import annotations

import logging
import time
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
    """

    requires_grid = True

    def __init__(
        self,
        fn: Callable[[float, float], float | None],
        sample_points: int = 5,
        delay: float = 0.05,
    ) -> None:
        self._fn = fn
        self._sample_points = max(1, sample_points)
        self._delay = delay

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
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
