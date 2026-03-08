# Data Sources — Overview

All data sources inherit from `DataSource` and implement a single method:

```python
def fetch(self, grid: GridSpec | None = None) -> RasterData:
    ...
```

`RasterData` is a named tuple `(array: np.ndarray, crs: str | None, transform: Affine | None)`.
No rasterio objects are ever exposed outside a source module.

## Source catalogue

| Source | Use case | Extra required |
|--------|----------|---------------|
| [`ArraySource`](core.md) | In-memory numpy array | — |
| [`ConstantSource`](core.md) | Single scalar broadcast over the grid | — |
| [`RasterSource`](core.md) | Local GeoTIFF file | `[io]` |
| [`URLSource`](core.md) | Remote GeoTIFF via HTTP | `[io]` |
| [`WCSSource`](wcs.md) | Generic OGC WCS endpoint | `[io]` |
| [`PointGridSource`](core.md) | Sample any callable over an N×N lat/lon grid | — |

## Grid-aware vs self-contained sources

**Self-contained** sources (`ArraySource`, `ConstantSource`, `RasterSource`, `URLSource`)
ignore the `grid` argument. They carry their own spatial metadata.

**Grid-aware** sources need the reference grid to determine what geographic area to
query. They call `grid.extent_wgs84()` to obtain `(lon_min, lat_min, lon_max, lat_max)`
before making API or WCS requests.

Grid-aware sources are: `WCSSource`, `PointGridSource`.

## DataSource ABC

::: geobn.sources.DataSource
    options:
      members:
        - fetch
