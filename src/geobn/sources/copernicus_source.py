"""Copernicus Marine Service (CMEMS) ocean data source."""
from __future__ import annotations

import os

import numpy as np
from affine import Affine

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource


class CopernicusMarineSource(DataSource):
    """Fetch ocean data from the Copernicus Marine Service (CMEMS).

    Requires a free CMEMS account.  Credentials are read from the constructor
    arguments, falling back to the environment variables
    ``COPERNICUSMARINE_SERVICE_USERNAME`` and
    ``COPERNICUSMARINE_SERVICE_PASSWORD``.

    Requires the ``copernicusmarine`` package::

        pip install geobn[ocean]

    Parameters
    ----------
    dataset_id:
        CMEMS dataset identifier, e.g.
        ``"cmems_mod_nws_phy-uv_anfc_0.083deg_PT1H-i"``.
    variable:
        Variable name within the dataset, e.g. ``"uo"`` (eastward current).
    datetime:
        ISO datetime string ``"YYYY-MM-DDTHH:MM:SS"`` for the desired
        time step.
    username:
        CMEMS username (falls back to env var).
    password:
        CMEMS password (falls back to env var).
    """

    requires_grid = True

    def __init__(
        self,
        dataset_id: str,
        variable: str,
        datetime: str,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._dataset_id = dataset_id
        self._variable = variable
        self._datetime = datetime
        self._username = username or os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
        self._password = password or os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        if not self._username or not self._password:
            raise ValueError(
                "CMEMS credentials are required.  Provide username/password "
                "in the constructor or set the environment variables "
                "COPERNICUSMARINE_SERVICE_USERNAME and "
                "COPERNICUSMARINE_SERVICE_PASSWORD."
            )

        if grid is None:
            raise ValueError(
                "CopernicusMarineSource requires a grid context to determine "
                "the spatial domain.  This is provided automatically by "
                "GeoBayesianNetwork.infer()."
            )

        try:
            import copernicusmarine as cm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "copernicusmarine is required for CopernicusMarineSource. "
                "Install it with: pip install geobn[ocean]"
            ) from exc

        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()

        try:
            ds = cm.open_dataset(
                dataset_id=self._dataset_id,
                username=self._username,
                password=self._password,
                minimum_longitude=lon_min,
                maximum_longitude=lon_max,
                minimum_latitude=lat_min,
                maximum_latitude=lat_max,
                start_datetime=self._datetime,
                end_datetime=self._datetime,
                variables=[self._variable],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open CMEMS dataset {self._dataset_id!r}: {exc}"
            ) from exc

        try:
            da = ds[self._variable]
            if "time" in da.dims:
                da = da.isel(time=0)
        except KeyError as exc:
            available = list(ds.data_vars)
            raise RuntimeError(
                f"Variable {self._variable!r} not found in dataset "
                f"{self._dataset_id!r}.  Available: {available}"
            ) from exc

        if "depth" in da.dims:
            da = da.isel(depth=0)

        array = da.values.astype(np.float32)

        # xarray may return south-to-north; flip to north-to-south (raster convention)
        lat_coords = da.coords.get("latitude", da.coords.get("lat"))
        if lat_coords is not None:
            lats = lat_coords.values
            if len(lats) > 1 and lats[0] < lats[-1]:
                array = np.flipud(array)
                lats = lats[::-1]

        lon_coords = da.coords.get("longitude", da.coords.get("lon"))
        lons = lon_coords.values if lon_coords is not None else None
        lats_vals = lats if lat_coords is not None else None

        if lons is not None and lats_vals is not None:
            pixel_w = (lons[-1] - lons[0]) / max(len(lons) - 1, 1)
            pixel_h = abs(lats_vals[0] - lats_vals[-1]) / max(len(lats_vals) - 1, 1)
            transform = Affine(
                pixel_w, 0, lons[0] - pixel_w / 2,
                0, -pixel_h, lats_vals[0] + pixel_h / 2,
            )
        else:
            pixel_w = (lon_max - lon_min) / max(array.shape[1] - 1, 1)
            pixel_h = (lat_max - lat_min) / max(array.shape[0] - 1, 1)
            transform = Affine(
                pixel_w, 0, lon_min - pixel_w / 2,
                0, -pixel_h, lat_max + pixel_h / 2,
            )

        return RasterData(array=array, crs="EPSG:4326", transform=transform)
