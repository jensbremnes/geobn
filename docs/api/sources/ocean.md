# Ocean & AIS Sources

These sources provide oceanographic model data and vessel tracking information.

## Credentials and extras

`CopernicusMarineSource` and `HubOceanSource` require the `[ocean]` extra:

```bash
pip install "geobn[ocean]"
```

They also require credentials:

- **Copernicus Marine**: register at [marine.copernicus.eu](https://marine.copernicus.eu/)
  and set `COPERNICUSMARINE_SERVICE_USERNAME` / `COPERNICUSMARINE_SERVICE_PASSWORD`
  environment variables (or use `~/.copernicusmarine/credentials`).
- **HubOcean**: set `HUBOCEAN_API_KEY` environment variable.

`BarentswatchAISSource` does not require an extra package but needs Barentswatch OAuth2
credentials (`BARENTSWATCH_CLIENT_ID` and `BARENTSWATCH_CLIENT_SECRET`).

---

## CopernicusMarineSource

::: geobn.CopernicusMarineSource
    options:
      show_root_heading: true

Fetches data from the [Copernicus Marine Service (CMEMS)](https://marine.copernicus.eu/)
using the official `copernicusmarine` SDK. Supports any CMEMS dataset and variable.

Credential and grid validation are performed **before** the lazy import, so
missing-credential errors are raised even when `copernicusmarine` is not installed.

---

## HubOceanSource

::: geobn.HubOceanSource
    options:
      show_root_heading: true

Fetches data from the [HubOcean](https://hubocean.earth/) STAC catalog using
`pystac_client` + `xarray`. Supports any dataset accessible via the HubOcean API.

---

## BarentswatchAISSource

::: geobn.BarentswatchAISSource
    options:
      show_root_heading: true

Fetches vessel positions from the [Barentswatch AIS API](https://www.barentswatch.no/en/bwapi/)
using OAuth2 client-credentials flow. Vessel lat/lon coordinates are rasterised to the
reference grid.

Available metrics (selected via the `metric` constructor argument):

| Metric | Description |
|--------|-------------|
| `"density"` | Vessels per km² (default) |
| `"count"` | Raw vessel count per pixel |
| `"speed"` | Mean vessel speed (knots) |

**Example:**

```python
import os

source = geobn.BarentswatchAISSource(
    client_id=os.environ["BARENTSWATCH_CLIENT_ID"],
    client_secret=os.environ["BARENTSWATCH_CLIENT_SECRET"],
    metric="density",
)
bn.set_input("vessel_density", source)
```
