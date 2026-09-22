from __future__ import annotations

from pathlib import Path

import rasterio

from .._io import read_first_band
from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource


class RasterSource(DataSource):
    """Read a local GeoTIFF file.

    rasterio is used only to open the file and is discarded immediately;
    the returned RasterData contains only plain numpy/affine objects.
    Pixels matching the file's declared nodata value (or its mask) are
    returned as NaN.

    Parameters
    ----------
    path:
        Path to the GeoTIFF file.
    valid_range:
        Optional ``(lo, hi)`` tuple.  Values outside this range become NaN.
        Use it for files that encode missing data as an extreme number
        without declaring it as nodata.  Either bound may be ``None``.
    """

    def __init__(
        self,
        path: str | Path,
        valid_range: tuple[float | None, float | None] | None = None,
    ) -> None:
        super().__init__(valid_range=valid_range)
        self._path = Path(path)

    def _fetch(self, grid: GridSpec | None = None) -> RasterData:
        with rasterio.open(self._path) as src:
            return read_first_band(src)
