"""Generic OGC Web Coverage Service (WCS) source."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import requests

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource


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
    extra_subsets:
        Additional ``SUBSET=`` values appended to the WCS 2.0 request (e.g.
        ``['time("2023-01-01T00:00:00.000Z")']`` for time-aware coverages).
        Ignored for WCS 1.x requests.
    """

    def __init__(
        self,
        url: str,
        layer: str,
        version: str = "2.0.1",
        format: str = "image/tiff",
        timeout: int = 60,
        cache_dir: str | Path | None = None,
        extra_subsets: list[str] | None = None,
    ) -> None:
        self._url = url
        self._layer = layer
        self._version = version
        self._format = format
        self._timeout = timeout
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self._extra_subsets = extra_subsets or []

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        try:
            from rasterio.io import MemoryFile  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "rasterio is required for WCSSource. "
                "Install it with: pip install geobn[io]"
            ) from exc

        if grid is None:
            raise ValueError(
                "WCSSource requires a grid context to determine the spatial "
                "domain.  This is provided automatically by "
                "GeoBayesianNetwork.infer()."
            )

        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        H, W = grid.shape

        # ── Cache check ───────────────────────────────────────────────────
        if self._cache_dir is not None:
            from ._cache import _load_cached, _make_cache_path, _save_cached  # noqa: PLC0415
            cache_key = {
                "url": self._url, "layer": self._layer, "version": self._version,
                "lon_min": round(lon_min, 8), "lat_min": round(lat_min, 8),
                "lon_max": round(lon_max, 8), "lat_max": round(lat_max, 8),
                "H": H, "W": W,
            }
            cache_path = _make_cache_path(self._cache_dir, cache_key)
            cached = _load_cached(cache_path)
            if cached is not None:
                return cached

        if self._version.startswith("2"):
            params = self._build_params_v2(lon_min, lat_min, lon_max, lat_max)
        elif self._version.startswith("1.0"):
            params = self._build_params_v0(lon_min, lat_min, lon_max, lat_max, H, W)
        else:
            params = self._build_params_v1(lon_min, lat_min, lon_max, lat_max)

        response = requests.get(self._url, params=params, timeout=self._timeout)
        if not response.ok:
            raise RuntimeError(
                f"WCS request failed with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                array = src.read(1).astype(np.float32)
                crs = src.crs.to_string()
                transform = src.transform

        result = RasterData(array=array, crs=crs, transform=transform)

        # ── Save to cache ─────────────────────────────────────────────────
        if self._cache_dir is not None:
            _save_cached(cache_path, result)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_params_v2(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
    ) -> dict:
        subsets = [
            f"Lat({lat_min},{lat_max})",
            f"Long({lon_min},{lon_max})",
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
