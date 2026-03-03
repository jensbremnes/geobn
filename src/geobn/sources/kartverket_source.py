"""Kartverket (Norwegian Mapping Authority) Digital Terrain Model source."""
from __future__ import annotations

import numpy as np

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource
from .wcs_source import WCSSource

# hoydedata.no ArcGIS WCS endpoints (WCS 1.0.0)
_LAYER_MAP = {
    "dtm10": (
        "https://hoydedata.no/arcgis/services/las_dtm_somlos/ImageServer/WCSServer",
        "las_dtm",
    ),
    "dom10": (
        "https://hoydedata.no/arcgis/services/las_dom_somlos/ImageServer/WCSServer",
        "las_dom",
    ),
}


class KartverketDTMSource(DataSource):
    """Fetch the Norwegian Digital Terrain Model from Kartverket's free WCS.

    Coverage is limited to Norway; pixels outside the valid bounds are
    returned as NaN (no exception is raised).

    Requires ``rasterio`` (``pip install geobn[io]``).

    Parameters
    ----------
    layer:
        Coverage type:

        * ``"dtm10"`` — seamless LiDAR terrain model (default).
        * ``"dom10"`` — seamless LiDAR surface model (includes vegetation/buildings).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(self, layer: str = "dtm10", timeout: int = 60) -> None:
        if layer not in _LAYER_MAP:
            raise ValueError(
                f"Unknown Kartverket layer {layer!r}. "
                f"Valid options: {list(_LAYER_MAP)}"
            )
        self._layer = layer
        self._timeout = timeout
        wcs_url, wcs_coverage = _LAYER_MAP[layer]
        self._wcs = WCSSource(
            url=wcs_url,
            layer=wcs_coverage,
            version="1.0.0",
            format="GeoTIFF",
            timeout=timeout,
        )

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        data = self._wcs.fetch(grid=grid)
        array = data.array.copy()

        # Replace sentinel nodata values with NaN
        array[(array < -500) | (array > 9000)] = np.nan

        return RasterData(array=array, crs=data.crs, transform=data.transform)
