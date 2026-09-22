# Data Sources — Overview

All data sources inherit from `DataSource` and implement a single method:

```python
def _fetch(self, grid: GridSpec | None = None) -> RasterData:
    ...
```

`RasterData` is a named tuple `(array: np.ndarray, crs: str | None, transform: Affine | None)`.
No rasterio objects are ever exposed outside a source module.

Callers use `fetch()`, which the base class provides. It reads and writes the disk cache,
calls `_fetch()` and applies `valid_range`. A source becomes cacheable by also implementing
`_cache_key(grid)`, which returns a JSON-serialisable dict identifying the request, or
`None` when the source is not cached.

## Source catalogue

| Source | Use case |
|--------|----------|
| [`ArraySource`](core.md) | In-memory numpy array |
| [`ConstantSource`](core.md) | Single scalar broadcast over the grid |
| [`RasterSource`](core.md) | Local GeoTIFF file |
| [`URLSource`](core.md) | Remote GeoTIFF via HTTP |
| [`WCSSource`](wcs.md) | Generic OGC WCS endpoint |
| [`PointGridSource`](core.md) | Sample any callable over an N×N lat/lon grid |

## Grid-aware vs self-contained sources

**Self-contained** sources (`ArraySource`, `ConstantSource`, `RasterSource`, `URLSource`)
ignore the `grid` argument. They carry their own spatial metadata.

**Grid-aware** sources need the reference grid to determine what geographic area to
query. They call `grid.extent_wgs84()` to obtain `(lon_min, lat_min, lon_max, lat_max)`
before making API or WCS requests. The box is computed from points along all four grid
edges, so it covers edges that curve in lon/lat, and it reaches the pole for polar grids.
A grid crossing the antimeridian gets a box spanning nearly all longitudes.

Grid-aware sources are: `WCSSource`, `PointGridSource`.

## Masking sentinel values with `valid_range`

Every source takes `valid_range=(lo, hi)`. Values below `lo` or above `hi` are replaced
with NaN, so they are excluded from inference instead of being discretised as real values.
Both bounds are inclusive, and either may be `None` to leave that side unbounded:

```python
geobn.RasterSource("dem.tif", valid_range=(-500.0, 9000.0))  # mask both tails
geobn.RasterSource("dem.tif", valid_range=(0.0, None))       # mask negatives only
```

The three GeoTIFF-backed sources (`RasterSource`, `URLSource`, `WCSSource`) convert a
declared nodata value and any mask band to NaN on their own. `valid_range` covers the data
that declares nothing and encodes missing values as an extreme number, such as −9999 or
32767.

For the cached sources the range is applied to the array on its way out, so the cache holds
the data as it arrived and changing the range does not trigger a new request.

## Caching and freshness

`URLSource`, `WCSSource` and `PointGridSource` take `cache_dir` to store what they fetched
on disk, and `cache_ttl` — a `timedelta` or a number of seconds — for how long an entry
stays usable. Without a TTL an entry is used forever, which is what terrain and bathymetry
want; with one, an older entry is fetched again.

Each entry is an `.npy` array beside a `.json` sidecar holding its CRS, transform and the
time it was fetched. `cache_ttl` is not part of the cache key, so changing it re-reads the
same entry rather than orphaning it.

If an entry has expired and the fetch then fails, the expired entry is returned with a
`UserWarning` naming its age, so an offline run still produces a map.

`PointGridSource` additionally needs `name`, since the callable it wraps cannot identify
itself.

## DataSource ABC

::: geobn.sources.DataSource
    options:
      members:
        - fetch
