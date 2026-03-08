# Generic WCS Source

## WCSSource

::: geobn.WCSSource
    options:
      show_root_heading: true

`WCSSource` is a generic [OGC Web Coverage Service](https://www.ogc.org/standard/wcs/)
client. It supports WCS 2.0.1, 1.1.1, and 1.0.0.

Requires `pip install "geobn[io]"`.

### Request construction

**WCS 2.0.1** uses `SUBSET` parameters:

```
...&SUBSET=Lat(lat_min,lat_max)&SUBSET=Long(lon_min,lon_max)
```

**WCS 1.1.1** uses a `BBOX` parameter.

**WCS 1.0.0** uses `COVERAGE`, `BBOX`, `WIDTH`, and `HEIGHT` parameters
(common for ArcGIS Image Server WCS endpoints).

### Nodata masking with `valid_range`

Pass `valid_range=(lo, hi)` to replace out-of-range sentinel values with NaN
after fetching. This is the standard way to handle services that encode nodata
as extreme numbers rather than a proper nodata band.

### Disk caching

Pass `cache_dir` to cache responses to disk. The cache key is a SHA-256 hash of the
request URL and parameters. Corrupt or missing cache entries trigger a fresh request.

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
