"""rasterio-backed I/O helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

from ._types import RasterData


def read_first_band(dataset: rasterio.io.DatasetReader) -> RasterData:
    """Read band 1 of an open rasterio dataset as float32 RasterData.

    Pixels flagged as missing by the dataset — its declared nodata value,
    an internal mask band, or alpha — are returned as NaN so they are
    excluded from inference instead of being binned as real values.
    """
    band = dataset.read(1, masked=True)
    array = band.astype(np.float32).filled(np.nan)
    return RasterData(array=array, crs=dataset.crs.to_string(), transform=dataset.transform)


def write_geotiff(
    array: np.ndarray,
    crs: str,
    transform: Affine,
    path: str | Path,
    nodata: float = float("nan"),
    descriptions: list[str] | None = None,
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
    descriptions:
        Optional band descriptions, one per band.
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
        if descriptions is not None:
            for i, desc in enumerate(descriptions, start=1):
                dst.set_band_description(i, desc)
