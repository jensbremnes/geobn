# Maritime Norway — Navigation Risk

**Location:** Outer coast between Stad and Ålesund, western Norway (62°N–63°N, 4.8°E–6.5°E)

This example demonstrates **multi-output** Bayesian inference from exclusively Norwegian
and European open data — no credentials required.

## What it demonstrates

- Two query nodes (`grounding_risk`, `collision_risk`) in a single `infer()` call
- Combining WCS bathymetry with live MET Norway API forecasts
- Cached static sources (EMODnet bathymetry + shipping density)
- Deriving a spatial binary indicator (wind-onshore) from scalar + raster inputs

## Bayesian network structure

```
wave_height   ─┐
               ├─► sea_state ─┬─► grounding_risk
current_speed  ┘              │         ↑
                              └─► collision_risk
                                        ↑
water_depth  ──────────────── grounding_risk
wind_onshore ──────────────── grounding_risk
vessel_density ─────────────── collision_risk
wind_speed   ──────────────── collision_risk
```

Six root nodes, one intermediate node (`sea_state`), two query nodes
(`grounding_risk`, `collision_risk`) each with states `{low, medium, high}`.

## Data sources

| Node | Source | Notes |
|------|--------|-------|
| `water_depth` | `EMODnetBathymetrySource` | positive depth (m); NaN on land |
| `wind_onshore` | derived from `METLocationForecastSource` | binary spatial indicator |
| `wave_height` | `METOceanForecastSource` | metres |
| `current_speed` | `METOceanForecastSource` | m/s |
| `wind_speed` | `METLocationForecastSource` | m/s |
| `vessel_density` | `EMODnetShippingDensitySource` | vessel hours/km²/month |

## Annotated walkthrough

### 1. Reference grid (100 × 170 pixels at ~1 km)

```python
WEST, SOUTH, EAST, NORTH = 4.8, 62.0, 6.5, 63.0
CRS = "EPSG:4326"
RESOLUTION = 0.01  # degrees (~1 km at this latitude)
```

### 2. Bathymetry → water depth

```python
bathy = geobn.EMODnetBathymetrySource(cache_dir=CACHE_DIR).fetch(grid=ref_grid)
# EMODnet returns negative values for ocean depth, positive for land
depth_m = np.where(bathy.array < 0, -bathy.array, np.nan).astype(np.float32)
```

### 3. Wind direction → wind_onshore spatial indicator

The `wind_onshore` node is a binary indicator (1.0 where wind blows toward shallower
water). It is derived from:

1. A mean wind direction fetched from MET Norway LocationForecast
2. The bathymetry gradient (direction toward shallower water at each pixel)

```python
wind_dir_data = geobn.METLocationForecastSource("wind_from_direction", sample_points=3).fetch(grid=ref_grid)
mean_wind_from = float(np.nanmean(wind_dir_data.array))
wind_onshore_arr = compute_wind_onshore(depth_m, mean_wind_from)
```

A fallback to 270° (prevailing westerly) is used if the MET Norway API is unavailable.

### 4. Wire all inputs

```python
bn.set_input("water_depth",    geobn.ArraySource(depth_m, ...))
bn.set_input("wind_onshore",   geobn.ArraySource(wind_onshore_arr, ...))
bn.set_input("wave_height",    geobn.METOceanForecastSource("sea_surface_wave_height", 5))
bn.set_input("current_speed",  geobn.METOceanForecastSource("sea_water_speed", 5))
bn.set_input("wind_speed",     geobn.METLocationForecastSource("wind_speed", 5))
bn.set_input("vessel_density", geobn.EMODnetShippingDensitySource(ship_type="all", year=2022, cache_dir=CACHE_DIR))
```

### 5. Discretization

```python
bn.set_discretization("wave_height",    [0, 0.5, 2.0, 10.0],   ["calm", "moderate", "rough"])
bn.set_discretization("current_speed",  [0, 0.3, 1.0, 5.0],    ["low", "moderate", "high"])
bn.set_discretization("wind_speed",     [0, 5, 15, 50],         ["calm", "moderate", "strong"])
bn.set_discretization("water_depth",    [0, 20, 100, 9999],     ["very_shallow", "shallow", "deep"])
bn.set_discretization("wind_onshore",   [0.0, 0.5, 1.5],        ["offshore", "onshore"])
bn.set_discretization("vessel_density", [0, 0.5, 5.0, 1_000_000], ["sparse", "moderate", "dense"])
```

### 6. Multi-output inference

```python
result = bn.infer(query=["grounding_risk", "collision_risk"])
result.to_geotiff(OUT_DIR)  # writes grounding_risk.tif and collision_risk.tif
```

## Key outputs

- **`output/grounding_risk.tif`** — 4-band GeoTIFF (P(low), P(medium), P(high), entropy)
- **`output/collision_risk.tif`** — 4-band GeoTIFF (P(low), P(medium), P(high), entropy)
- **`output/map.html`** — interactive Leaflet map showing both risk layers + water depth

Land pixels are NaN in all outputs.

## How to run

```bash
uv run python examples/maritime_norway/run_example.py
```

The first run fetches EMODnet bathymetry and shipping density and saves them to
`examples/maritime_norway/cache/`. MET Norway forecasts are fetched live on every run
(they are not cached as they change daily).
