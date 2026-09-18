# Core sources

## ArraySource

::: geobn.ArraySource
    options:
      show_root_heading: true

With CRS metadata (fully georeferenced):

```python
import numpy as np
from affine import Affine

slope = np.random.rand(100, 200).astype("float32") * 45.0
transform = Affine(0.01, 0, 10.0, 0, -0.01, 70.0)
source = geobn.ArraySource(slope, crs="EPSG:4326", transform=transform)
```

A pre-aligned array, without CRS:

```python
# Slope computed from a DEM fetched with fetch_raw() is already on the
# BN grid, so it can be passed without CRS metadata.
dem = bn.fetch_raw(geobn.WCSSource(...))
slope_deg = compute_slope(dem)   # same shape as the BN grid
bn.set_input("slope_angle", geobn.ArraySource(slope_deg))
```

## ConstantSource

::: geobn.ConstantSource
    options:
      show_root_heading: true

```python
# 30 cm of recent snow everywhere
source = geobn.ConstantSource(30.0)
bn.set_input("recent_snow", source)
```

## RasterSource

::: geobn.RasterSource
    options:
      show_root_heading: true

```python
source = geobn.RasterSource("dem_10m.tif")
bn.set_input("elevation", source)
```

## URLSource

::: geobn.URLSource
    options:
      show_root_heading: true

Pass `cache_dir` to cache downloads on disk:

```python
source = geobn.URLSource(
    "https://example.com/data/slope.tif",
    cache_dir="cache/",
)
bn.set_input("slope_angle", source)
```

## PointGridSource

::: geobn.PointGridSource
    options:
      show_root_heading: true

`PointGridSource` works with any data you can query by point. Give it a callable
that takes `(lat, lon)` and returns a float.

Open-Meteo precipitation:

```python
import requests
import geobn

def fetch_precipitation(lat: float, lon: float) -> float:
    """Query Open-Meteo historical archive for daily precipitation."""
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
            "daily": "precipitation_sum",
            "timezone": "UTC",
        },
        timeout=10,
    )
    resp.raise_for_status()
    values = resp.json().get("daily", {}).get("precipitation_sum", [None])
    v = values[0]
    return float(v) if v is not None else float("nan")

source = geobn.PointGridSource(fn=fetch_precipitation, sample_points=5)
bn.set_input("precipitation", source)
```

MET Norway ocean forecast:

```python
import requests

def sea_temperature(lat: float, lon: float) -> float:
    resp = requests.get(
        "https://api.met.no/weatherapi/oceanforecast/2.0/complete",
        params={"lat": lat, "lon": lon},
        headers={"User-Agent": "my-app/1.0"},
        timeout=10,
    )
    if resp.status_code == 422:
        return float("nan")  # outside ocean coverage
    resp.raise_for_status()
    ts = resp.json().get("properties", {}).get("timeseries", [])
    if not ts:
        return float("nan")
    val = ts[0]["data"]["instant"]["details"].get("sea_water_temperature")
    return float(val) if val is not None else float("nan")

source = geobn.PointGridSource(fn=sea_temperature, sample_points=5, delay=0.1)
bn.set_input("sea_temp", source)
```
