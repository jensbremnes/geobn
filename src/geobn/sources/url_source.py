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
    cache_ttl:
        Maximum age of a cache entry, as a :class:`~datetime.timedelta` or a
        number of seconds.  An older entry is refetched.  The default
        ``None`` never expires.
    valid_range:
        Optional ``(lo, hi)`` tuple.  Values outside this range become NaN.
        Use it for files that encode missing data as an extreme number
        without declaring it as nodata.  Either bound may be ``None``.  The
        cached array is the one the server sent, so changing the range
        re-masks it without downloading again.
    """

    def __init__(
        self,
        url: str,
        timeout: int = 60,
        cache_dir: str | Path | None = None,
        valid_range: tuple[float | None, float | None] | None = None,
        cache_ttl: timedelta | float | None = None,
    ) -> None:
        super().__init__(
            valid_range=valid_range, cache_dir=cache_dir, cache_ttl=cache_ttl
        )
        self._url = url
        self._timeout = timeout

    def _cache_key(self, grid: GridSpec | None = None) -> dict | None:
        return {"url": self._url}

    def _fetch(self, grid: GridSpec | None = None) -> RasterData:
        _log.info("Fetching %s", self._url)
        response = requests.get(self._url, timeout=self._timeout)
        response.raise_for_status()
        _log.info("Downloaded: %.0f KB", len(response.content) / 1024)

        with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                return read_first_band(src)
