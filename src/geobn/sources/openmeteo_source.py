from __future__ import annotations

import logging
import time
from datetime import date as _date

import numpy as np
import requests

_log = logging.getLogger(__name__)
from affine import Affine

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource

_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_API = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoSource(DataSource):
    """Fetch a daily weather variable from the Open-Meteo API.

    The bounding box is derived at *fetch()* time from the reference grid,
    so no explicit coordinates are required in the constructor.

    A regular grid of *sample_points* × *sample_points* lat/lon points is
    queried and the resulting values are assembled into a coarse raster
    (EPSG:4326).  The alignment step in *GeoBayesianNetwork.infer()* then
    bilinearly resamples this to the reference grid resolution.

    Parameters
    ----------
    variable:
        Open-Meteo daily variable name, e.g. "precipitation_sum",
        "temperature_2m_mean".
    date:
        ISO date string "YYYY-MM-DD".  Defaults to today.
    sample_points:
        Number of sample points along each axis (total = sample_points²).
        Defaults to 5 (25 API calls).
    timeout:
        HTTP request timeout in seconds.
    """

    requires_grid = True

    def __init__(
        self,
        variable: str,
        date: str | None = None,
        sample_points: int = 5,
        timeout: int = 10,
    ) -> None:
        self._variable = variable
        self._date = date or str(_date.today())
        self._sample_points = max(1, sample_points)
        self._timeout = timeout

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        if grid is None:
            raise ValueError(
                "OpenMeteoSource requires a grid context to determine the "
                "spatial domain.  This is provided automatically by "
                "GeoBayesianNetwork.infer()."
            )

        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        n = self._sample_points

        _log.info(
            "Open-Meteo: sampling %d×%d grid (%d API calls), variable='%s', date=%s",
            n, n, n * n, self._variable, self._date,
        )

        lats = np.linspace(lat_max, lat_min, n)  # north → south (row order)
        lons = np.linspace(lon_min, lon_max, n)
        lon_grid, lat_grid = np.meshgrid(lons, lats)  # (n, n) each

        values = np.full((n, n), np.nan, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                val = self._query_point(float(lat_grid[i, j]), float(lon_grid[i, j]))
                values[i, j] = val
                if n > 1:
                    time.sleep(0.05)  # be polite to the free API

        valid_values = values[~np.isnan(values)]
        if valid_values.size > 0:
            _log.info(
                "Open-Meteo: done — values range %.2f–%.2f", valid_values.min(), valid_values.max()
            )
        else:
            _log.info("Open-Meteo: done — all values are NaN")

        if n == 1:
            # Single-point result — return as a ConstantSource-style 1×1 array
            return RasterData(array=values, crs=None, transform=None)

        pixel_h = (lat_max - lat_min) / (n - 1)
        pixel_w = (lon_max - lon_min) / (n - 1)
        transform = Affine(pixel_w, 0, lon_min - pixel_w / 2, 0, -pixel_h, lat_max + pixel_h / 2)

        return RasterData(array=values, crs="EPSG:4326", transform=transform)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_point(self, lat: float, lon: float) -> float:
        api = _ARCHIVE_API
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": self._variable,
            "timezone": "UTC",
            "start_date": self._date,
            "end_date": self._date,
        }
        resp = requests.get(api, params=params, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        values = daily.get(self._variable)
        if not values:
            raise ValueError(
                f"Open-Meteo returned no data for variable '{self._variable}' "
                f"on {self._date} at lat={lat:.4f}, lon={lon:.4f}.  "
                f"Check the variable name and date range."
            )
        return float(values[0]) if values[0] is not None else float("nan")
