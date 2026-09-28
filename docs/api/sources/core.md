# Core Sources

## ArraySource

::: geobn.ArraySource
    options:
      show_root_heading: true

**Example — with CRS metadata (full georeferencing):**

```python
import numpy as np
from affine import Affine

slope = np.random.rand(100, 200).astype("float32") * 45.0
transform = Affine(0.01, 0, 10.0, 0, -0.01, 70.0)
source = geobn.ArraySource(slope, crs="EPSG:4326", transform=transform)
```

**Example — pre-aligned array (no CRS needed):**

```python
# An array computed on the BN grid elsewhere, with the grid's shape,
# needs no CRS metadata.
snow_depth = my_snow_model(bn.fetch_raw(geobn.RasterSource("dem.tif")))
bn.set_input("snow_depth", geobn.ArraySource(snow_depth))
```

To compute an input from another source, a `DerivedSource` is usually the better fit:
it keeps the input lazy, so it is fetched only when needed and can be frozen.

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

**Example:**

```python
source = geobn.RasterSource("dem_10m.tif")
bn.set_input("elevation", source)
```

**Example — a class raster:**

```python
# Most frequent class per grid pixel, so no in-between codes appear
source = geobn.RasterSource("land_cover.tif", resampling="mode")
bn.set_input("land_cover", source)
```

**Example — masking undeclared sentinel values:**

```python
# A DTM that writes -9999 for sea without declaring it as nodata
source = geobn.RasterSource("dem_10m.tif", valid_range=(-500.0, 9000.0))
```

---

## URLSource

::: geobn.URLSource
    options:
      show_root_heading: true

Supports optional disk caching.

**Example:**

```python
source = geobn.URLSource(
    "https://example.com/data/slope.tif",
    cache_dir="cache/",
)
bn.set_input("slope_angle", source)
```

Add `cache_ttl` (a `timedelta` or seconds) for a file that is updated; an entry older than
the TTL is requested again. If the server sent an `ETag` or `Last-Modified` header with the
file, that request is conditional: a `304 Not Modified` answer keeps the cached array and
resets its age, and only a changed file is downloaded.

---

## PointGridSource

::: geobn.PointGridSource
    options:
      show_root_heading: true

`PointGridSource` is the generic primitive for any point-queryable data source.
Pass any callable that accepts `(lat, lon)` and returns a float.

**Caching a forecast:** pass `cache_dir` together with `name`, and `cache_ttl` for how long
a sampled lattice stays usable. On a hit `fn` is not called at all, which matters when each
call is an HTTP request — a 5×5 lattice is 25 of them.

`name` is what identifies the entry, because a callable cannot identify itself: two sources
built by the same factory are indistinguishable, so they would otherwise share one entry.
Passing `cache_dir` without `name` raises `ValueError`.

```python
from datetime import timedelta

source = geobn.PointGridSource(
    fn=sea_temperature,
    sample_points=5,
    name="sea_temp",
    cache_dir="cache/",
    cache_ttl=timedelta(hours=6),
)
```

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

---

## MosaicSource

::: geobn.MosaicSource
    options:
      show_root_heading: true

Each source is aligned to the reference grid before the merge, so they may differ in CRS,
resolution and extent. Each source is resampled with its own `resampling` method, and the
seam between two sources follows the resampled footprint of the higher-priority one. Sources are fetched in order and only
while pixels remain uncovered, so a remote source at the end of the list costs nothing
when the sources above it already cover the grid.

**Example — a local survey over a regional model:**

```python
depth = geobn.MosaicSource(
    [
        geobn.RasterSource("multibeam_survey.tif"),
        geobn.WCSSource(url=EMODNET_WCS, layer="emodnet:mean"),
    ],
    names=["survey", "emodnet"],
)
bn.set_input("water_depth", depth)
```

**Example — reading the provenance layer:**

```python
values, provenance = bn.fetch_raw(depth, return_provenance=True)

for index, name in enumerate(depth.names):
    print(f"{name}: {(provenance == index).mean():.0%} of the grid")
print(f"no data: {(provenance == -1).mean():.0%}")
```

**Example — a prior wherever the raster has nothing:**

```python
# ConstantSource covers the whole grid, so it is the natural last entry.
traffic = geobn.MosaicSource(
    [geobn.RasterSource("ais_density.tif"), geobn.ConstantSource(2.0)],
    names=["ais_density", "medium_traffic_prior"],
    on_error="skip",
)
```

With `on_error="skip"` a source that cannot be fetched at all — a file that is not there,
an endpoint that is down — counts as having no data, and a `UserWarning` names it. The
default `on_error="raise"` propagates the error, so a mistyped path is not covered up by
the source below it.

A `valid_range` on the mosaic applies to the merged values, after a pixel has been taken
from a source. A `valid_range` on one of the sources applies before the merge, and
therefore lets the next source fill in the pixels it masks.

---

## DerivedSource

::: geobn.DerivedSource
    options:
      show_root_heading: true

**Example — elevation to depth:**

```python
def depth_below_surface(elevation):
    return np.where(elevation <= 0, -elevation, np.nan)   # land → NaN

depth = geobn.DerivedSource(
    depth_below_surface,
    geobn.WCSSource(url=EMODNET_WCS, layer="emodnet:mean", cache_dir="cache/"),
)
bn.set_input("water_depth", depth)
```

**Example — combining two sources:**

```python
# Snow load from depth (m) and density (kg/m³); both are aligned to the grid first.
load = geobn.DerivedSource(
    lambda depth, density: depth * density * 9.81 / 1000,   # kPa
    geobn.RasterSource("snow_depth.tif"),
    geobn.URLSource("https://example.com/snow_density.tif"),
)
bn.set_input("snow_load", load)
```

**Example — building on a terrain helper:**

```python
def facing_north(aspect_deg):
    return np.where(np.isnan(aspect_deg), np.nan, np.cos(np.radians(aspect_deg)))

bn.set_input("northness", geobn.DerivedSource(facing_north, geobn.terrain.aspect(dem)))
```

The function receives float32 arrays with NaN for missing data and must return an array
of the grid's shape. The inputs are resampled with their own `resampling` method. A derived
source keeps no disk cache of its own, so give remote inputs a `cache_dir`; the function
runs again on every fetch, and `bn.freeze()` keeps its discretised result between
`infer()` calls.
