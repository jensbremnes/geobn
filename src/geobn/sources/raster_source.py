from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource


class RasterSource(DataSource):
    """Read a local GeoTIFF file.

    rasterio is used only to open the file and is discarded immediately;
    the returned RasterData contains only plain numpy/affine objects.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        with rasterio.open(self._path) as src:
            array = src.read(1).astype(np.float32)
            crs = src.crs.to_string()
            transform = src.transform  # affine.Affine — safe to keep

        return RasterData(array=array, crs=crs, transform=transform)
