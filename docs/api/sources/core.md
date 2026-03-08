# Core Sources

These sources are always available — no optional extras required (except `RasterSource`
and `URLSource` which need `[io]`).

## ArraySource

::: geobn.ArraySource
    options:
      show_root_heading: true

**Example:**

```python
import numpy as np
from affine import Affine

slope = np.random.rand(100, 200).astype("float32") * 45.0
transform = Affine(0.01, 0, 10.0, 0, -0.01, 70.0)
source = geobn.ArraySource(slope, crs="EPSG:4326", transform=transform)
```

---

## ConstantSource

::: geobn.ConstantSource
    options:
      show_root_heading: true

**Example:**

```python
# Apply a uniform 30 cm recent snowfall across the entire domain
source = geobn.ConstantSource(30.0)
bn.set_input("recent_snow", source)
```

---

## RasterSource

::: geobn.RasterSource
    options:
      show_root_heading: true

Requires `pip install "geobn[io]"`.

**Example:**

```python
source = geobn.RasterSource("dem_10m.tif", band=1)
bn.set_input("elevation", source)
```

---

## URLSource

::: geobn.URLSource
    options:
      show_root_heading: true

Requires `pip install "geobn[io]"`. Supports optional disk caching.

**Example:**

```python
source = geobn.URLSource(
    "https://example.com/data/slope.tif",
    cache_dir="cache/",
)
bn.set_input("slope_angle", source)
```

---

## PointGridSource

::: geobn.PointGridSource
    options:
      show_root_heading: true

`PointGridSource` is the generic primitive for any point-queryable data source.
Pass any callable that accepts `(lat, lon)` and returns a float.

**Example — Open-Meteo precipitation:**

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

**Example — MET Norway ocean forecast:**

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
