from __future__ import annotations

import numpy as np
from affine import Affine

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource


class ArraySource(DataSource):
    """Wrap a numpy array that is already in memory.

    This is the primary integration point for QGIS and other environments
    that manage rasters internally and cannot rely on file I/O.

    If *crs* and *transform* are omitted, the array is treated as
    pre-aligned to the BN's grid — ``align_to_grid`` will return it as-is
    when its shape matches the target grid, just like ``ConstantSource``
    broadcasts when it has no spatial metadata.
    """

    def __init__(
        self,
        array: np.ndarray,
        crs: str | None = None,
        transform: Affine | None = None,
    ) -> None:
        if array.ndim != 2:
            raise ValueError(f"array must be 2-D (H, W), got shape {array.shape}")
        self._array = array.astype(np.float32)
        self._crs = crs
        self._transform = transform

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        return RasterData(array=self._array, crs=self._crs, transform=self._transform)
