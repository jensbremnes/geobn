"""MET Norway API sources — ocean and atmospheric forecasts."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import requests
from affine import Affine

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource

_USER_AGENT = "geobn/0.1"

_OCEAN_VARIABLES: frozenset[str] = frozenset(
    {
        "sea_water_temperature",
        "sea_surface_wave_height",
        "sea_water_speed",
        "sea_surface_wave_from_direction",
        "sea_surface_wave_period",
    }
)

_LOCATION_VARIABLES: frozenset[str] = frozenset(
    {
        "air_temperature",
        "wind_speed",
        "wind_from_direction",
        "precipitation_amount",
        "relative_humidity",
        "cloud_area_fraction",
    }
)

# precipitation_amount comes from next_1_hours.details, not instant.details
_PRECIPITATION_VARIABLE = "precipitation_amount"


class METOceanForecastSource(DataSource):
    """Fetch ocean forecast data from the MET Norway OceanForecast 2.0 API.

    A regular grid of *sample_points* × *sample_points* lat/lon points is
    queried and assembled into a coarse raster (EPSG:4326).  Points outside
    ocean coverage return HTTP 422 and are filled with NaN.

    Parameters
    ----------
    variable:
        One of ``sea_water_temperature``, ``sea_surface_wave_height``,
        ``sea_water_speed``, ``sea_surface_wave_from_direction``,
        ``sea_surface_wave_period``.
    offset_hours:
        Hours from now to target.  The closest available forecast step is used.
    sample_points:
        Number of sample points along each axis (total = sample_points²).
    timeout:
        HTTP request timeout in seconds.
    """

    requires_grid = True
    _API = "https://api.met.no/weatherapi/oceanforecast/2.0/complete"

    def __init__(
        self,
        variable: str,
        offset_hours: int = 0,
        sample_points: int = 5,
        timeout: int = 10,
    ) -> None:
        if variable not in _OCEAN_VARIABLES:
            raise ValueError(
                f"Unknown ocean variable {variable!r}. "
                f"Valid options: {sorted(_OCEAN_VARIABLES)}"
            )
        self._variable = variable
        self._offset_hours = offset_hours
        self._sample_points = max(1, sample_points)
        self._timeout = timeout

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        if grid is None:
            raise ValueError(
                "METOceanForecastSource requires a grid context to determine "
                "the spatial domain.  This is provided automatically by "
                "GeoBayesianNetwork.infer()."
            )

        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        n = self._sample_points

        lats = np.linspace(lat_max, lat_min, n)
        lons = np.linspace(lon_min, lon_max, n)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        values = np.full((n, n), np.nan, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                val = self._query_point(float(lat_grid[i, j]), float(lon_grid[i, j]))
                values[i, j] = val
                if n > 1:
                    time.sleep(0.05)

        if n == 1:
            return RasterData(array=values, crs=None, transform=None)

        pixel_h = (lat_max - lat_min) / (n - 1)
        pixel_w = (lon_max - lon_min) / (n - 1)
        transform = Affine(pixel_w, 0, lon_min - pixel_w / 2, 0, -pixel_h, lat_max + pixel_h / 2)
        return RasterData(array=values, crs="EPSG:4326", transform=transform)

    def _query_point(self, lat: float, lon: float) -> float:
        headers = {"User-Agent": _USER_AGENT}
        params = {"lat": lat, "lon": lon}
        resp = requests.get(self._API, params=params, headers=headers, timeout=self._timeout)

        if resp.status_code == 422:
            # Outside ocean coverage
            return float("nan")
        resp.raise_for_status()

        data = resp.json()
        timeseries = data.get("properties", {}).get("timeseries", [])
        if not timeseries:
            return float("nan")

        target_dt = datetime.now(tz=timezone.utc).timestamp() + self._offset_hours * 3600
        best_entry = min(
            timeseries,
            key=lambda e: abs(
                datetime.fromisoformat(e["time"].replace("Z", "+00:00")).timestamp()
                - target_dt
            ),
        )

        details = (
            best_entry.get("data", {})
            .get("instant", {})
            .get("details", {})
        )
        val = details.get(self._variable)
        return float(val) if val is not None else float("nan")


class METLocationForecastSource(DataSource):
    """Fetch atmospheric forecast data from the MET Norway LocationForecast 2.0 API.

    A regular grid of *sample_points* × *sample_points* lat/lon points is
    queried and assembled into a coarse raster (EPSG:4326).

    Parameters
    ----------
    variable:
        One of ``air_temperature``, ``wind_speed``, ``wind_from_direction``,
        ``precipitation_amount``, ``relative_humidity``, ``cloud_area_fraction``.
    offset_hours:
        Hours from now to target.  The closest available forecast step is used.
    sample_points:
        Number of sample points along each axis (total = sample_points²).
    timeout:
        HTTP request timeout in seconds.
    """

    requires_grid = True
    _API = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

    def __init__(
        self,
        variable: str,
        offset_hours: int = 0,
        sample_points: int = 5,
        timeout: int = 10,
    ) -> None:
        if variable not in _LOCATION_VARIABLES:
            raise ValueError(
                f"Unknown location variable {variable!r}. "
                f"Valid options: {sorted(_LOCATION_VARIABLES)}"
            )
        self._variable = variable
        self._offset_hours = offset_hours
        self._sample_points = max(1, sample_points)
        self._timeout = timeout

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        if grid is None:
            raise ValueError(
                "METLocationForecastSource requires a grid context to determine "
                "the spatial domain.  This is provided automatically by "
                "GeoBayesianNetwork.infer()."
            )

        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        n = self._sample_points

        lats = np.linspace(lat_max, lat_min, n)
        lons = np.linspace(lon_min, lon_max, n)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        values = np.full((n, n), np.nan, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                val = self._query_point(float(lat_grid[i, j]), float(lon_grid[i, j]))
                values[i, j] = val
                if n > 1:
                    time.sleep(0.05)

        if n == 1:
            return RasterData(array=values, crs=None, transform=None)

        pixel_h = (lat_max - lat_min) / (n - 1)
        pixel_w = (lon_max - lon_min) / (n - 1)
        transform = Affine(pixel_w, 0, lon_min - pixel_w / 2, 0, -pixel_h, lat_max + pixel_h / 2)
        return RasterData(array=values, crs="EPSG:4326", transform=transform)

    def _query_point(self, lat: float, lon: float) -> float:
        headers = {"User-Agent": _USER_AGENT}
        params = {"lat": lat, "lon": lon}
        resp = requests.get(self._API, params=params, headers=headers, timeout=self._timeout)
        resp.raise_for_status()

        data = resp.json()
        timeseries = data.get("properties", {}).get("timeseries", [])
        if not timeseries:
            return float("nan")

        target_dt = datetime.now(tz=timezone.utc).timestamp() + self._offset_hours * 3600
        best_entry = min(
            timeseries,
            key=lambda e: abs(
                datetime.fromisoformat(e["time"].replace("Z", "+00:00")).timestamp()
                - target_dt
            ),
        )

        entry_data = best_entry.get("data", {})

        if self._variable == _PRECIPITATION_VARIABLE:
            val = (
                entry_data.get("next_1_hours", {})
                .get("details", {})
                .get(self._variable)
            )
        else:
            val = (
                entry_data.get("instant", {})
                .get("details", {})
                .get(self._variable)
            )

        return float(val) if val is not None else float("nan")
