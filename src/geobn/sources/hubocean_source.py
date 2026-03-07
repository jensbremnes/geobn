"""HubOcean ocean data platform source (STAC + xarray)."""
from __future__ import annotations

import os

import numpy as np
from affine import Affine

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource

_CATALOG_URL = "https://catalog.hubocean.earth"


class HubOceanSource(DataSource):
    """Fetch ocean data from the HubOcean data platform via STAC + xarray.

    Requires a HubOcean API key (set ``HUBOCEAN_API_KEY`` env var or pass
    via the constructor) and the ``pystac-client`` package::

        pip install geobn[ocean]

    Parameters
    ----------
    dataset_id:
        STAC collection ID on the HubOcean catalog.
    variable:
        Variable name in the dataset's xarray representation.
    datetime:
        Optional ISO datetime or interval (e.g. ``"2024-01-01T00:00:00Z"``).
        Passed directly to the STAC search.
    api_key:
        HubOcean API key (falls back to ``HUBOCEAN_API_KEY`` env var).
    """

    requires_grid = True

    def __init__(
        self,
        dataset_id: str,
        variable: str,
        datetime: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._dataset_id = dataset_id
        self._variable = variable
        self._datetime = datetime
        self._api_key = api_key or os.environ.get("HUBOCEAN_API_KEY")

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        if not self._api_key:
            raise ValueError(
                "A HubOcean API key is required.  Provide it via the "
                "api_key constructor argument or set the HUBOCEAN_API_KEY "
                "environment variable."
            )

        if grid is None:
            raise ValueError(
                "HubOceanSource requires a grid context to determine the "
                "spatial domain.  This is provided automatically by "
                "GeoBayesianNetwork.infer()."
            )

        try:
            import pystac_client  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pystac-client is required for HubOceanSource. "
                "Install it with: pip install geobn[ocean]"
            ) from exc

        try:
            import xarray as xr  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "xarray is required for HubOceanSource. "
                "Install it with: pip install geobn[ocean]"
            ) from exc

        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        bbox = [lon_min, lat_min, lon_max, lat_max]

        headers = {"Authorization": f"Bearer {self._api_key}"}
        catalog = pystac_client.Client.open(_CATALOG_URL, headers=headers)

        search_kwargs: dict = {
            "collections": [self._dataset_id],
            "bbox": bbox,
        }
        if self._datetime:
            search_kwargs["datetime"] = self._datetime

        search = catalog.search(**search_kwargs)
        items = list(search.items())

        if not items:
            raise ValueError(
                f"No STAC items found for collection {self._dataset_id!r} "
                f"within bbox {bbox} (datetime={self._datetime!r})."
            )

        # Use the first item's data asset
        item = items[0]
        asset = None
        for key in ("data", "zarr", "netcdf", "nc"):
            if key in item.assets:
                asset = item.assets[key]
                break
        if asset is None:
            # Fall back to first asset
            asset = next(iter(item.assets.values()))

        try:
            ds = xr.open_dataset(asset.href)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open dataset from {asset.href!r}: {exc}"
            ) from exc

        if self._variable not in ds:
            available = list(ds.data_vars)
            raise ValueError(
                f"Variable {self._variable!r} not found in dataset. "
                f"Available variables: {available}"
            )

        da = ds[self._variable]

        if "time" in da.dims:
            da = da.isel(time=0)
        if "depth" in da.dims:
            da = da.isel(depth=0)

        array = da.values.astype(np.float32)

        lat_coords = da.coords.get("latitude", da.coords.get("lat"))
        lon_coords = da.coords.get("longitude", da.coords.get("lon"))

        if lat_coords is not None:
            lats = lat_coords.values
            if len(lats) > 1 and lats[0] < lats[-1]:
                array = np.flipud(array)
                lats = lats[::-1]
        else:
            lats = None

        if lat_coords is not None and lon_coords is not None:
            lons = lon_coords.values
            pixel_w = (lons[-1] - lons[0]) / max(len(lons) - 1, 1)
            pixel_h = abs(lats[0] - lats[-1]) / max(len(lats) - 1, 1)
            transform = Affine(
                pixel_w, 0, lons[0] - pixel_w / 2,
                0, -pixel_h, lats[0] + pixel_h / 2,
            )
        else:
            pixel_w = (lon_max - lon_min) / max(array.shape[1] - 1, 1)
            pixel_h = (lat_max - lat_min) / max(array.shape[0] - 1, 1)
            transform = Affine(
                pixel_w, 0, lon_min - pixel_w / 2,
                0, -pixel_h, lat_max + pixel_h / 2,
            )

        return RasterData(array=array, crs="EPSG:4326", transform=transform)
