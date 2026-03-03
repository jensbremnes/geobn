# Generic WCS Source

## WCSSource

::: geobn.WCSSource
    options:
      show_root_heading: true

`WCSSource` is a generic [OGC Web Coverage Service](https://www.ogc.org/standard/wcs/)
client. It supports WCS 2.0.1 (primary) with automatic fallback to WCS 1.1.1.

Requires `pip install "geobn[io]"`.

### Request construction

**WCS 2.0.1** uses `SUBSET` parameters:

```
...&SUBSET=Lat(lat_min,lat_max)&SUBSET=Long(lon_min,lon_max)
```

**WCS 1.1.1 fallback** uses a `BBOX` parameter. The fallback is triggered automatically
if the 2.0.1 request fails.

### Disk caching

Pass `cache_dir` to cache responses to disk. The cache key is a SHA-256 hash of the
request URL and parameters. Corrupt or missing cache entries trigger a fresh request.

### Usage

```python
source = geobn.WCSSource(
    url="https://wcs.example.com/wcs",
    layer="my_coverage",
    crs="EPSG:4326",
    cache_dir="cache/",
)
data = source.fetch(grid=ref_grid)
```

### Composing WCSSource

`KartverketDTMSource`, `EMODnetBathymetrySource`, and `EMODnetShippingDensitySource`
are thin wrappers around `WCSSource` that hard-code the endpoint URL, coverage name, and
nodata sentinel logic. You can follow the same pattern to add new WCS-based sources:

```python
from geobn.sources.wcs_source import WCSSource
from geobn.sources._base import DataSource
from geobn._types import RasterData
import numpy as np

class MyWCSSource(DataSource):
    def __init__(self, cache_dir=None):
        self._wcs = WCSSource(
            url="https://my-wcs-server.com/wcs",
            layer="my_coverage",
            crs="EPSG:4326",
            cache_dir=cache_dir,
        )

    def fetch(self, grid=None):
        data = self._wcs.fetch(grid=grid)
        arr = data.array.copy()
        arr[arr < -9999] = np.nan  # apply nodata sentinel
        return RasterData(arr, data.crs, data.transform)
```
