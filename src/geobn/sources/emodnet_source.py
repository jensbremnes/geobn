"""EMODnet (European Marine Observation and Data Network) data sources.

Provides:
- ``EMODnetBathymetrySource`` — European seabed depth from the EMODnet
  Bathymetry WCS (https://ows.emodnet-bathymetry.eu/wcs).
- ``EMODnetShippingDensitySource`` — Historical vessel traffic density from
  the EMODnet Human Activities WCS
  (https://www.emodnet-humanactivities.eu/geoserver/emodnet/wcs).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource
from .wcs_source import WCSSource

_WCS_BATHYMETRY = "https://ows.emodnet-bathymetry.eu/wcs"
_WCS_HUMAN_ACTIVITIES = "https://ows.emodnet-humanactivities.eu/wcs"

# Coverage IDs on the new WCS 2.0.1 endpoint (annual averages).
# Source: GetCapabilities as of 2025.
_SHIPPING_LAYER_MAP: dict[str, str] = {
    "all":      "emodnet__vesseldensity_allavg",
    "cargo":    "emodnet__vesseldensity_09avg",
    "tanker":   "emodnet__vesseldensity_10avg",
    "fishing":  "emodnet__vesseldensity_01avg",
    "passenger": "emodnet__vesseldensity_08avg",
    "highspeed": "emodnet__vesseldensity_06avg",
    "sailing":  "emodnet__vesseldensity_04avg",
    "pleasure": "emodnet__vesseldensity_05avg",
    "service":  "emodnet__vesseldensity_02avg",
    "dredging": "emodnet__vesseldensity_03avg",
    "tug":      "emodnet__vesseldensity_07avg",
    "military": "emodnet__vesseldensity_11avg",
    "unknown":  "emodnet__vesseldensity_12avg",
    "other":    "emodnet__vesseldensity_00avg",
}


class EMODnetBathymetrySource(DataSource):
    """Fetch European seabed bathymetry from the EMODnet bathymetry WCS.

    Returns depth values as **negative floats** (e.g. −200 m means 200 m
    below sea level).  Pixels outside EMODnet's coverage (land areas) are
    filled with NaN.

    Requires ``rasterio`` (``pip install geobn[io]``).

    Parameters
    ----------
    layer:
        Coverage to fetch:

        * ``"emodnet:mean"`` — mean depth (default).
        * ``"emodnet:stdev"`` — depth uncertainty (standard deviation).
    timeout:
        HTTP request timeout in seconds.
    cache_dir:
        Optional path to a directory for caching fetched rasters on disk.
        On a cache hit the HTTP request is skipped entirely.
    """

    _VALID_LAYERS = frozenset({"emodnet:mean", "emodnet:stdev"})

    def __init__(
        self,
        layer: str = "emodnet:mean",
        timeout: int = 60,
        cache_dir: str | Path | None = None,
    ) -> None:
        if layer not in self._VALID_LAYERS:
            raise ValueError(
                f"Unknown EMODnet layer {layer!r}. "
                f"Valid options: {sorted(self._VALID_LAYERS)}"
            )
        self._layer = layer
        self._timeout = timeout
        self._wcs = WCSSource(
            url=_WCS_BATHYMETRY,
            layer=layer,
            version="2.0.1",
            format="image/tiff",
            timeout=timeout,
            cache_dir=cache_dir,
        )

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        data = self._wcs.fetch(grid=grid)
        array = data.array.copy()

        # Replace sentinel nodata values with NaN
        # Depths > 9000 m or shallower than −15 000 m are clearly invalid
        array[(array > 9000) | (array < -15000)] = np.nan

        return RasterData(array=array, crs=data.crs, transform=data.transform)


class EMODnetShippingDensitySource(DataSource):
    """Fetch historical vessel traffic density from the EMODnet Human Activities WCS.

    Returns gridded vessel hours per km² per month, derived from satellite AIS
    data.  Coverage: European waters.  Free, no authentication required.

    Requires ``rasterio`` (``pip install geobn[io]``).

    Parameters
    ----------
    ship_type:
        Vessel category.  One of ``"all"``, ``"cargo"``, ``"tanker"``,
        ``"fishing"``, ``"passenger"``, ``"highspeed"``, ``"sailing"``,
        ``"pleasure"``, ``"service"``, ``"dredging"``, ``"tug"``,
        ``"military"``, ``"unknown"``, ``"other"``.
    year:
        Year of the annual average snapshot (2017–2023, default 2023).
    timeout:
        HTTP request timeout in seconds.
    cache_dir:
        Optional path to a directory for caching fetched rasters on disk.
        On a cache hit the HTTP request is skipped entirely.
    """

    _VALID_SHIP_TYPES = frozenset(_SHIPPING_LAYER_MAP)
    _VALID_YEARS = range(2017, 2024)

    def __init__(
        self,
        ship_type: str = "all",
        year: int = 2023,
        timeout: int = 60,
        cache_dir: str | Path | None = None,
    ) -> None:
        if ship_type not in self._VALID_SHIP_TYPES:
            raise ValueError(
                f"Unknown ship_type {ship_type!r}. "
                f"Valid options: {sorted(self._VALID_SHIP_TYPES)}"
            )
        if year not in self._VALID_YEARS:
            raise ValueError(
                f"year {year} is out of range. "
                f"Valid years: {self._VALID_YEARS.start}–{self._VALID_YEARS.stop - 1}"
            )
        self._ship_type = ship_type
        self._year = year
        self._timeout = timeout
        layer = _SHIPPING_LAYER_MAP[ship_type]
        self._wcs = WCSSource(
            url=_WCS_HUMAN_ACTIVITIES,
            layer=layer,
            version="2.0.1",
            format="image/tiff",
            timeout=timeout,
            cache_dir=cache_dir,
            extra_subsets=[f'time("{year}-01-01T00:00:00.000Z")'],
        )

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        data = self._wcs.fetch(grid=grid)
        array = data.array.copy()

        # Replace nodata sentinels: negative values and extreme outliers are invalid.
        # Vessel density is vessel hours/km²/month, so valid range is [0, ~1e6).
        array[(array < 0) | (array > 1e6)] = np.nan

        return RasterData(array=array, crs=data.crs, transform=data.transform)
