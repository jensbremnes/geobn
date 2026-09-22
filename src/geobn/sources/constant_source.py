from __future__ import annotations

import numpy as np

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource


class ConstantSource(DataSource):
    """Broadcast a single scalar value across the entire domain.

    The 1×1 sentinel array is recognised by *align_to_grid* and expanded
    to the reference grid shape, so no explicit spatial metadata is needed.

    Parameters
    ----------
    value:
        The scalar to broadcast.
    valid_range:
        Optional ``(lo, hi)`` tuple.  Either bound may be ``None``.  A value
        outside the range becomes NaN, and so covers the whole grid with
        NoData.
    """

    def __init__(
        self,
        value: float,
        valid_range: tuple[float | None, float | None] | None = None,
    ) -> None:
        super().__init__(valid_range=valid_range)
        self._value = float(value)

    def _fetch(self, grid: GridSpec | None = None) -> RasterData:
        return RasterData(
            array=np.array([[self._value]], dtype=np.float32),
            crs=None,
            transform=None,
        )
