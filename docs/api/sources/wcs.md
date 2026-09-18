# WCS source

## WCSSource

::: geobn.WCSSource
    options:
      show_root_heading: true

`WCSSource` is a generic [OGC Web Coverage Service](https://www.ogc.org/standard/wcs/)
client. It supports WCS 2.0.1, 1.1.1, and 1.0.0.

### Request construction

**WCS 2.0.1** uses `SUBSET` parameters:

```
...&SUBSET=Lat(lat_min,lat_max)&SUBSET=Long(lon_min,lon_max)
```

The bounds are in EPSG:4326 (`SUBSETTINGCRS`). The default axis labels are `Lat`/`Long`.
Servers that name the axes differently (e.g. `lat`/`lon` or `y`/`x`) reject
the request, so pass `axis_labels=(lon_label, lat_label)` to match the server:

```python
source = geobn.WCSSource(url, layer="my_coverage", axis_labels=("lon", "lat"))
```

The axis names are listed in the server's DescribeCoverage response.
`extra_subsets` adds further subsets, e.g. `['time("2023-01-01T00:00:00.000Z")']`.

**WCS 1.1.1** uses a `BBOX` parameter.

**WCS 1.0.0** uses `COVERAGE`, `BBOX`, `WIDTH`, and `HEIGHT` parameters
(common for ArcGIS Image Server WCS endpoints).

### Nodata masking with `valid_range`

If the returned GeoTIFF declares a nodata value, those pixels are converted to NaN
automatically. Pass `valid_range=(lo, hi)` to additionally replace out-of-range
sentinel values with NaN. Some services encode nodata as extreme numbers without
declaring it, and need this.

### Disk caching

Pass `cache_dir` to cache responses to disk. The cache key is a SHA-256 hash of the
request URL and parameters (bounding box, output size, format, axis labels, extra
subsets and `valid_range`), plus an internal cache-format version. The version is
bumped when the meaning of cached content changes, so old entries are fetched again.
A corrupt or missing cache entry also triggers a new request.

### Recipes

Kartverket Norwegian DTM (10 m):

```python
source = geobn.WCSSource(
    url="https://hoydedata.no/arcgis/services/las_dtm_somlos/ImageServer/WCSServer",
    layer="las_dtm",
    version="1.0.0",
    format="GeoTIFF",
    valid_range=(-500.0, 9000.0),   # mask fill values outside elevation range
    cache_dir="cache/",
)
bn.set_input("elevation", source)
```

EMODnet European bathymetry:

```python
source = geobn.WCSSource(
    url="https://ows.emodnet-bathymetry.eu/wcs",
    layer="emodnet:mean",
    version="1.0.0",
    valid_range=(-15000.0, 9000.0),  # negative depths are valid; mask extreme sentinels
    cache_dir="cache/",
)
bn.set_input("depth", source)
```

EMODnet shipping density:

```python
source = geobn.WCSSource(
    url="https://ows.emodnet-humanactivities.eu/wcs",
    layer="emodnet:density_all_2024",
    version="2.0.1",
    valid_range=(0.0, 1_000_000.0),
    cache_dir="cache/",
)
bn.set_input("shipping_density", source)
```
