# Generic WCS Source

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

The bounds are in EPSG:4326 (`SUBSETTINGCRS`). `Lat`/`Long` are the default axis
labels; servers that name the axes differently (e.g. `lat`/`lon` or `y`/`x`) reject
this request. Pass `axis_labels=(lon_label, lat_label)` to match the server:

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
sentinel values with NaN — needed for services that encode nodata as extreme
numbers without declaring it. See
[Masking sentinel values](index.md#masking-sentinel-values-with-valid_range) for the
full description; it works the same on every source.

### Disk caching

Pass `cache_dir` to cache responses to disk. The cache key is a SHA-256 hash of the
request URL and parameters (bounding box, output size, format, axis labels and extra
subsets), plus an internal cache-format version (bumped when cached content semantics
change, so outdated entries are refetched). Corrupt or missing cache entries trigger a
fresh request. `valid_range` is not part of the key: the response is cached as it
arrived and the range is applied on the way out, so adjusting it costs no new request.

`cache_ttl` sets how long an entry stays usable, as a `timedelta` or a number of seconds.
It is not part of the key either — freshness is a property of the entry, not of what was
requested — so changing it never orphans a cached coverage. Leave it unset for terrain and
bathymetry, which do not change:

```python
source = geobn.WCSSource(
    url="https://example.com/wcs",
    layer="sea_surface_temperature",
    cache_ttl=timedelta(hours=3),
)
```

If the coverage has expired and the request then fails, the expired entry is returned with
a `UserWarning` naming its age.

### Recipes

**Kartverket Norwegian DTM (10 m):**

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

**EMODnet European Bathymetry:**

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

**EMODnet Shipping Density:**

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
