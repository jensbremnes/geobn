# Lyngen Alps — Avalanche Risk

**Location:** Lyngen Alps, Tromsø county, northern Norway (69.35°N–69.75°N, 19.8°E–21.0°E)

This example demonstrates pixel-wise avalanche risk inference over real Norwegian terrain
using a free WCS endpoint. No credentials are required.

## What it demonstrates

- Fetching a real 10 m Digital Terrain Model from Kartverket's WCS
- Deriving slope angle and aspect analytically from the DEM
- Using `ConstantSource` for spatially-uniform weather inputs
- Encoding domain knowledge (terrain + weather) in a 2-level BN
- Exploring different weather scenarios by changing two scalar constants

## Bayesian network structure

```
slope_angle ──┐
               ├──► terrain_factor ──┐
sun_exposure ──┘                     ├──► avalanche_risk
recent_snow ──┐                      │
               ├──► weather_factor ──┘
temperature ──┘
```

Four root nodes (evidence inputs), two intermediate nodes, one query node
(`avalanche_risk`) with states `{low, medium, high}`.

## Data sources

| Node | Source | Notes |
|------|--------|-------|
| `slope_angle` | `WCSSource` (Kartverket DTM) → numpy.gradient | degrees (0–90°) |
| `sun_exposure` | `WCSSource` (Kartverket DTM) → numpy.gradient | aspect quadrant (0=N, 1=E, 2=W, 3=S) |
| `recent_snow` | `ConstantSource` | cm; edit `RECENT_SNOW_CM` to change scenario |
| `temperature` | `ConstantSource` | °C; edit `AIR_TEMP_C` to change scenario |

## Annotated walkthrough

### 1. Define the study area and grid

```python
WEST, SOUTH, EAST, NORTH = 19.8, 69.35, 21.0, 69.75
CRS = "EPSG:4326"
RESOLUTION = 0.005   # ~200 m at 70°N  →  80 rows × 240 cols

H = round((NORTH - SOUTH) / RESOLUTION)   # 80
W = round((EAST  - WEST)  / RESOLUTION)   # 240
transform = Affine(RESOLUTION, 0, WEST, 0, -RESOLUTION, NORTH)
ref_grid = GridSpec(crs=CRS, transform=transform, shape=(H, W))
```

### 2. Fetch the DTM

```python
dem = bn.fetch_raw(geobn.WCSSource(
    url="https://hoydedata.no/arcgis/services/las_dtm_somlos/ImageServer/WCSServer",
    layer="las_dtm",
    version="1.0.0",
    valid_range=(-500.0, 9000.0),
    cache_dir=CACHE_DIR,
))  # (80, 240) float32, NaN at sea
```

The terrain is cached after the first run. On subsequent runs it loads from
`examples/lyngen_alps/cache/` without making a network request.

### 3. Derive slope and aspect

```python
slope_deg, north_facing = compute_slope_aspect(dem)
# slope_deg:    float32 (H, W), range 0–90°
# north_facing: float32 (H, W), 1.0 = N-facing, 0.0 = S-facing
```

The pixel-metre conversion accounts for the geographic CRS:

```python
m_per_deg_lat = 111_320.0
m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_mid))
dz_drow, dz_dcol = np.gradient(dem_filled, pixel_lat_m, pixel_lon_m)
```

### 4. Load the BN and wire inputs

```python
bn = geobn.load("avalanche_risk.bif")
bn.set_grid(CRS, RESOLUTION, (WEST, SOUTH, EAST, NORTH))

bn.set_input("slope_angle",  geobn.ArraySource(slope_deg))
bn.set_input("sun_exposure", geobn.ArraySource(sun_exposure))
bn.set_input("recent_snow", geobn.ConstantSource(RECENT_SNOW_CM))
bn.set_input("temperature",  geobn.ConstantSource(AIR_TEMP_C))
```

### 5. Configure discretization

```python
bn.set_discretization("slope_angle",  [0, 5, 25, 40, 90])
bn.set_discretization("sun_exposure", [-0.5, 0.5, 1.5, 2.5, 3.5])
bn.set_discretization("recent_snow",  [0, 10, 25, 150])
bn.set_discretization("temperature",  [-40, -8, -2, 15])
```

### 6. Run inference and export

```python
result = bn.infer(query=["avalanche_risk"])
result.show_map(OUT_DIR, extra_layers={"Slope angle (°)": slope_deg})
result.to_geotiff(OUT_DIR)
```

## Key outputs

- **`output/map.html`** — interactive Leaflet map with risk probability overlay,
  entropy overlay, slope angle overlay, and layer switcher
- **`output/avalanche_risk.tif`** — 4-band GeoTIFF:
  Band 1 P(low), Band 2 P(medium), Band 3 P(high), Band 4 entropy

## How to run

```bash
python examples/lyngen_alps/run_example.py
```

## Exploring weather scenarios

Edit the two constants at the top of `run_example.py`:

```python
RECENT_SNOW_CM = 30.0   # cm  — heavy recent snowfall
AIR_TEMP_C     = -5.0   # °C  — cold but not extreme
```

Try `RECENT_SNOW_CM = 5.0` (light snow) and `AIR_TEMP_C = 0.5` (warming/wet) to
see how the risk distribution changes across the terrain.
