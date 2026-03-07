from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .._types import RasterData

if TYPE_CHECKING:
    from ..grid import GridSpec


class DataSource(ABC):
    """Base class for all geographic data sources.

    Every source returns a plain RasterData tuple (array, crs, transform)
    — no rasterio objects are ever exposed outside the source module.
    """

    # Subclasses that require a grid bbox before they can fetch (e.g. WCS,
    # Open-Meteo) override this to True.  Self-contained sources (Raster,
    # Array, URL, Constant) keep the default False.
    requires_grid: bool = False

    @abstractmethod
    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        """Fetch data aligned to *grid* if provided.

        Parameters
        ----------
        grid:
            Reference grid context.  Required by sources that need to know the
            spatial domain before querying (e.g. OpenMeteoSource).  Ignored by
            sources that are self-contained (Array, Raster, URL, Constant).
        """
