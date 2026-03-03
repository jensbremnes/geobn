# Terrain & Bathymetry Sources

These sources fetch elevation and seabed data from WCS endpoints. All require
`pip install "geobn[io]"` (rasterio is used to decode WCS GeoTIFF responses).

Disk caching is supported via `cache_dir`: on first fetch the response is saved to
a `.npy` / `.json` sidecar pair keyed by a SHA-256 hash of the request parameters.
Subsequent fetches with the same parameters load from disk without hitting the network.

## KartverketDTMSource

::: geobn.KartverketDTMSource
    options:
      show_root_heading: true

Fetches Norway's free 10 m Digital Terrain Model from [Kartverket](https://www.geonorge.no/).
Coverage: mainland Norway and Svalbard only. Returns NaN for sea pixels and areas
outside coverage.

Nodata sentinel: values `< −500` or `> 9000` are replaced with NaN.

**Example:**

```python
source = geobn.KartverketDTMSource(cache_dir="cache/")
dtm_data = source.fetch(grid=ref_grid)
dem = dtm_data.array  # (H, W) float32, NaN at sea
```

---

## EMODnetBathymetrySource

::: geobn.EMODnetBathymetrySource
    options:
      show_root_heading: true

Fetches the [EMODnet Bathymetry](https://emodnet.ec.europa.eu/en/bathymetry) seabed
elevation grid (~115 m resolution). Covers European seas. No registration required.

Values are signed elevations: negative = below sea level (ocean), positive = above sea
level (land / intertidal). Nodata sentinel: values `> 9000` or `< −15000` → NaN.

**Example:**

```python
source = geobn.EMODnetBathymetrySource(cache_dir="cache/")
bathy = source.fetch(grid=ref_grid)
depth_m = np.where(bathy.array < 0, -bathy.array, np.nan)  # positive depth, NaN on land
```

---

## EMODnetShippingDensitySource

::: geobn.EMODnetShippingDensitySource
    options:
      show_root_heading: true

Fetches historical vessel traffic density (vessel hours / km² / month) from
[EMODnet Human Activities](https://www.emodnet-humanactivities.eu/). No registration
required. Covers European seas.

Nodata sentinel: values `< 0` or `> 1e6` → NaN.

**Example:**

```python
source = geobn.EMODnetShippingDensitySource(
    ship_type="all",
    year=2022,
    cache_dir="cache/",
)
density = source.fetch(grid=ref_grid)
```
