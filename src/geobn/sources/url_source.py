from __future__ import annotations

from pathlib import Path

import numpy as np
import requests

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource

# rasterio is imported lazily; URLSource therefore also requires geobn[io].


class URLSource(DataSource):
    """Fetch a GeoTIFF from an HTTP/HTTPS URL and return it as plain data.

    The file is streamed into an in-memory buffer so nothing is written to
    disk.  rasterio is the only component that sees the raw bytes; the
    returned RasterData contains only numpy / affine objects.

    Parameters
    ----------
    url:
        HTTP/HTTPS URL pointing to a GeoTIFF file.
    timeout:
        HTTP request timeout in seconds.
    cache_dir:
        Optional path to a directory for caching the fetched raster on disk.
        On a cache hit the HTTP request is skipped entirely.
    """

    def __init__(self, url: str, timeout: int = 60, cache_dir: str | Path | None = None) -> None:
        self._url = url
        self._timeout = timeout
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        try:
            import rasterio  # noqa: PLC0415
            from rasterio.io import MemoryFile  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "rasterio is required for URLSource. "
                "Install it with: pip install geobn[io]"
            ) from exc

        # ── Cache check ───────────────────────────────────────────────────
        if self._cache_dir is not None:
            from ._cache import _load_cached, _make_cache_path, _save_cached  # noqa: PLC0415
            cache_key = {"url": self._url}
            cache_path = _make_cache_path(self._cache_dir, cache_key)
            cached = _load_cached(cache_path)
            if cached is not None:
                return cached

        response = requests.get(self._url, timeout=self._timeout)
        response.raise_for_status()

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
