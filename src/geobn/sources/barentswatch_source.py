"""Barentswatch AIS vessel tracking source."""
from __future__ import annotations

import numpy as np
import requests
from affine import Affine
from pyproj import Transformer

from .._types import RasterData
from ..grid import GridSpec
from ._base import DataSource

_TOKEN_URL = "https://id.barentswatch.no/connect/token"
_AIS_URL = "https://live.ais.barentswatch.no/v1/combined"

_VALID_METRICS = frozenset({"density", "count", "speed"})


class BarentswatchAISSource(DataSource):
    """Rasterize live AIS vessel positions from Barentswatch onto the grid.

    Requires a free Barentswatch account with an API client (OAuth2
    client-credentials flow).  Register at `barentswatch.no
    <https://www.barentswatch.no>`_.

    Results outside the Norwegian economic zone will be an empty (all-zero)
    raster — this is not an error.

    Parameters
    ----------
    client_id:
        OAuth2 client ID from barentswatch.no.
    client_secret:
        OAuth2 client secret.
    vessel_types:
        Optional list of AIS vessel-type codes to filter on.  ``None`` means
        all types.
    metric:
        ``"density"`` — vessels per km²; ``"count"`` — raw vessel count per
        pixel; ``"speed"`` — mean speed over ground (knots) per pixel.
    timeout:
        HTTP request timeout in seconds.
    """

    requires_grid = True

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        vessel_types: list[int] | None = None,
        metric: str = "density",
        timeout: int = 30,
    ) -> None:
        if metric not in _VALID_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}. Valid options: {sorted(_VALID_METRICS)}"
            )
        self._client_id = client_id
        self._client_secret = client_secret
        self._vessel_types = set(vessel_types) if vessel_types else None
        self._metric = metric
        self._timeout = timeout

    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        if grid is None:
            raise ValueError(
                "BarentswatchAISSource requires a grid context to determine "
                "the spatial domain.  This is provided automatically by "
                "GeoBayesianNetwork.infer()."
            )

        token = self._get_token()
        vessels = self._fetch_vessels(token, grid)

        array = self._rasterize(vessels, grid)
        return RasterData(array=array, crs=grid.crs, transform=grid.transform)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        resp = requests.post(
            _TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "scope": "ais",
            },
            timeout=self._timeout,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Barentswatch token request failed with HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        return resp.json()["access_token"]

    def _fetch_vessels(self, token: str, grid: GridSpec) -> list[dict]:
        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "Xmin": lon_min,
            "Ymin": lat_min,
            "Xmax": lon_max,
            "Ymax": lat_max,
        }
        resp = requests.get(
            _AIS_URL, params=params, headers=headers, timeout=self._timeout
        )
        if not resp.ok:
            raise RuntimeError(
                f"Barentswatch AIS request failed with HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        vessels = resp.json()
        if self._vessel_types is not None:
            vessels = [v for v in vessels if v.get("shipType") in self._vessel_types]
        return vessels

    def _rasterize(self, vessels: list[dict], grid: GridSpec) -> np.ndarray:
        H, W = grid.shape
        count = np.zeros((H, W), dtype=np.float32)
        speed_sum = np.zeros((H, W), dtype=np.float32)

        if not vessels:
            if self._metric == "density":
                return count  # all zeros
            return count

        # Transform vessel WGS84 positions → grid CRS
        to_grid = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
        inv_transform = ~grid.transform

        for vessel in vessels:
            lat = vessel.get("latitude") or vessel.get("lat")
            lon = vessel.get("longitude") or vessel.get("lon")
            sog = vessel.get("speedOverGround") or vessel.get("sog") or 0.0

            if lat is None or lon is None:
                continue

            grid_x, grid_y = to_grid.transform(float(lon), float(lat))
            col_f, row_f = inv_transform * (grid_x, grid_y)
            col, row = int(col_f), int(row_f)

            if 0 <= row < H and 0 <= col < W:
                count[row, col] += 1
                speed_sum[row, col] += float(sog)

        if self._metric == "count":
            return count

        if self._metric == "speed":
            with np.errstate(invalid="ignore"):
                result = np.where(count > 0, speed_sum / count, np.nan)
            return result.astype(np.float32)

        # metric == "density": vessels per km²
        # Pixel area in km² (approximation using grid units)
        t = grid.transform
        pixel_area_units2 = abs(t.a * t.e)  # a=dx, e=-dy in map units
        try:
            from pyproj import CRS  # noqa: PLC0415
            crs_obj = CRS.from_user_input(grid.crs)
            unit = crs_obj.axis_info[0].unit_name
            if unit in ("metre", "meter"):
                pixel_area_km2 = pixel_area_units2 / 1e6
            elif unit in ("degree", "degree (supplier to define representation)"):
                # 1 degree² ≈ (111.32 km)²; approximate at mid-latitude
                pixel_area_km2 = pixel_area_units2 * (111.32 ** 2)
            else:
                raise ValueError(
                    f"Cannot compute vessel density for CRS with unit '{unit}'. "
                    "Use a metric CRS (e.g. EPSG:32633) or choose metric='count'."
                )
        except ValueError:
            raise
        except Exception:
            pixel_area_km2 = pixel_area_units2 / 1e6  # assume metres

        density = count / max(pixel_area_km2, 1e-12)
        return density.astype(np.float32)
