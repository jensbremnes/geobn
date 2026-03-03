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
