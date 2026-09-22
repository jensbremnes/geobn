"""Generic OGC Web Coverage Service (WCS) source."""
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import requests
from rasterio.io import MemoryFile

from .._io import read_first_band
from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource

_log = logging.getLogger(__name__)


class WCSSource(DataSource):
    """Fetch a raster band from any OGC Web Coverage Service.

    Builds a GetCoverage HTTP request for the grid bounding box, receives
    GeoTIFF bytes, and parses them via ``rasterio.MemoryFile`` (same
    approach as URLSource).

    Parameters
    ----------
    url:
        Base URL of the WCS service (no query parameters).
    layer:
        Coverage identifier (``COVERAGEID`` for 2.0, ``IDENTIFIER`` for 1.1).
    version:
        WCS version string — ``"2.0.1"`` (default) or ``"1.1.1"``.
    format:
        Output format MIME type (default ``"image/tiff"``).
    timeout:
        HTTP request timeout in seconds.
    cache_dir:
        Optional path to a directory for caching fetched rasters on disk.
        On a cache hit the HTTP request is skipped entirely.  Useful for
        static sources (terrain, bathymetry) where the data never changes.
    cache_ttl:
        Maximum age of a cache entry, as a :class:`~datetime.timedelta` or a
        number of seconds.  An older entry is refetched.  The default
        ``None`` never expires, which suits terrain and bathymetry; set it
        for coverages that are updated.
    extra_subsets:
        Additional ``SUBSET=`` values appended to the WCS 2.0 request (e.g.
        ``['time("2023-01-01T00:00:00.000Z")']`` for time-aware coverages).
        Ignored for WCS 1.x requests.
    axis_labels:
        Names of the longitude and latitude axes used in the WCS 2.0
        ``SUBSET=`` parameters, as ``(lon_label, lat_label)``.  The default
        ``("Long", "Lat")`` matches the EPSG:4326 axis abbreviations; some
        servers use e.g. ``("lon", "lat")`` or ``("x", "y")`` instead.  The
        bounds are always given in EPSG:4326 (``SUBSETTINGCRS``).  Ignored
        for WCS 1.x requests.
    valid_range:
        Optional ``(lo, hi)`` tuple.  Pixels outside this range are set to
        NaN.  Use this to mask out nodata sentinels that the server encodes
        as extreme numeric values (e.g. ``(-500, 9000)`` for Norwegian DTM
        data where values below −500 m or above 9000 m are fill values).
        Either bound may be ``None``.  The cached array is the one the
        server sent, so changing the range re-masks it without requesting
        the coverage again.
    """

    requires_grid = True

    def __init__(
        self,
        url: str,
        layer: str,
        version: str = "2.0.1",
        format: str = "image/tiff",
        timeout: int = 60,
        cache_dir: str | Path | None = None,
        extra_subsets: list[str] | None = None,
        valid_range: tuple[float | None, float | None] | None = None,
        axis_labels: tuple[str, str] = ("Long", "Lat"),
        cache_ttl: timedelta | float | None = None,
    ) -> None:
        super().__init__(
            valid_range=valid_range, cache_dir=cache_dir, cache_ttl=cache_ttl
        )
        if (
            isinstance(axis_labels, str)
            or len(axis_labels) != 2
            or not all(isinstance(a, str) and a for a in axis_labels)
        ):
            raise ValueError(
                "axis_labels must be two non-empty strings "
                f"(lon_label, lat_label); got {axis_labels!r}"
            )
        self._url = url
        self._layer = layer
        self._version = version
        self._format = format
        self._timeout = timeout
        self._extra_subsets = extra_subsets or []
        self._axis_labels = tuple(axis_labels)

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def _fetch(self, grid: GridSpec | None = None) -> RasterData:
        if grid is None:
            raise ValueError(
                "WCSSource requires a grid context to determine the spatial "
                "domain.  This is provided automatically by "
                "GeoBayesianNetwork.infer()."
            )

        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        H, W = grid.shape

        if self._version.startswith("2"):
            params = self._build_params_v2(lon_min, lat_min, lon_max, lat_max)
        elif self._version.startswith("1.0"):
            params = self._build_params_v0(lon_min, lat_min, lon_max, lat_max, H, W)
        else:
            params = self._build_params_v1(lon_min, lat_min, lon_max, lat_max)

        _log.info(
            "WCS fetch: %s, bbox=(%.4f, %.4f, %.4f, %.4f)",
            self._layer, lon_min, lat_min, lon_max, lat_max,
        )
        response = requests.get(self._url, params=params, timeout=self._timeout)
        if not response.ok:
            raise RuntimeError(
                f"WCS request failed with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        _log.info("WCS response: %.0f KB in %.1fs", len(response.content) / 1024, response.elapsed.total_seconds())

        with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                return read_first_band(src)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, grid: GridSpec | None = None) -> dict | None:
        if grid is None:
            return None
        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        H, W = grid.shape
        key = {
            "url": self._url, "layer": self._layer, "version": self._version,
            "lon_min": round(lon_min, 8), "lat_min": round(lat_min, 8),
            "lon_max": round(lon_max, 8), "lat_max": round(lat_max, 8),
            "H": H, "W": W,
        }
        # Options that change the request or the cached array.  Added only
        # when not at their default, so existing cache entries stay valid.
        if self._format != "image/tiff":
            key["format"] = self._format
        if self._axis_labels != ("Long", "Lat"):
            key["axis_labels"] = list(self._axis_labels)
        if self._extra_subsets:
            key["extra_subsets"] = list(self._extra_subsets)
        return key

    def _build_params_v2(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
    ) -> dict:
        lon_label, lat_label = self._axis_labels
        subsets = [
            f"{lat_label}({lat_min},{lat_max})",
            f"{lon_label}({lon_min},{lon_max})",
        ] + list(self._extra_subsets)
        return {
            "SERVICE": "WCS",
            "VERSION": self._version,
            "REQUEST": "GetCoverage",
            "COVERAGEID": self._layer,
            "FORMAT": self._format,
            "SUBSET": subsets,
            "SUBSETTINGCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
        }

    def _build_params_v0(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        height: int,
        width: int,
    ) -> dict:
        """WCS 1.0.0 GetCoverage parameters.

        Uses ``COVERAGE`` (not ``IDENTIFIER``) and ``WIDTH``/``HEIGHT`` for
        output dimensions.
        """
        return {
            "SERVICE": "WCS",
            "VERSION": self._version,
            "REQUEST": "GetCoverage",
            "COVERAGE": self._layer,
            "FORMAT": self._format,
            "BBOX": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "CRS": "EPSG:4326",
            "WIDTH": width,
            "HEIGHT": height,
        }

    def _build_params_v1(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
    ) -> dict:
        return {
            "SERVICE": "WCS",
            "VERSION": self._version,
            "REQUEST": "GetCoverage",
            "IDENTIFIER": self._layer,
            "FORMAT": self._format,
            "BBOX": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "CRS": "EPSG:4326",
        }
