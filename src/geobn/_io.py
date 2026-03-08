"""rasterio-backed I/O helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS


def write_geotiff(
    array: np.ndarray,
    crs: str,
    transform: Affine,
    path: str | Path,
    nodata: float = float("nan"),
) -> None:
    """Write a multi-band float32 GeoTIFF.

    Parameters
    ----------
    array:
        (bands, H, W) float32 array.
    crs:
        CRS as EPSG string or WKT.
    transform:
        Affine pixel-to-world transform.
    path:
        Output file path.
    nodata:
        NoData value written into the file metadata.
    """
    path = Path(path)
    bands, H, W = array.shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=bands,
        dtype=np.float32,
        crs=CRS.from_user_input(crs),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array.astype(np.float32))
