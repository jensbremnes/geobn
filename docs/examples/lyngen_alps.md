# Lyngen Alps — Avalanche Risk

**Location:** Lyngen Alps, Tromsø county, northern Norway (69.35°N–69.75°N, 19.8°E–21.0°E)

This example demonstrates pixel-wise avalanche risk inference over real Norwegian terrain
using a free WCS endpoint. No credentials are required.

## What it demonstrates

- Fetching a real 10 m Digital Terrain Model from Kartverket's WCS
- Deriving slope angle and aspect from the DEM with `geobn.terrain` and `DerivedSource`
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
| `slope_angle` | `terrain.slope` of the Kartverket DTM (`WCSSource`) | degrees (0–90°) |
| `sun_exposure` | `terrain.aspect` of the DTM, classified by a `DerivedSource` | aspect quadrant (0=N, 1=E, 2=W, 3=S) |
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

### 2. Describe the DTM and the terrain inputs

```python
def mask_sea(dem):
    return np.where(dem > 0, dem, np.nan)   # Kartverket returns 0 for sea

dem = geobn.DerivedSource(mask_sea, geobn.WCSSource(
    url="https://hoydedata.no/arcgis/services/las_dtm_somlos/ImageServer/WCSServer",
    layer="las_dtm",
    version="1.0.0",
    format="GeoTIFF",
    valid_range=(-500.0, 9000.0),
    cache_dir=CACHE_DIR,
))
slope = geobn.terrain.slope(dem)
sun_exposure = geobn.DerivedSource(aspect_quadrant, geobn.terrain.aspect(dem))
forest_cover = geobn.DerivedSource(treeline_forest_cover, dem)
```

Nothing is fetched yet. The DTM is downloaded when the first of these inputs is needed,
and cached in `examples/lyngen_alps/cache/`, so later fetches and later runs read it from
disk.

`geobn.terrain.slope` and `geobn.terrain.aspect` measure the distance between pixel
centres on the ellipsoid, so a 0.005° pixel at 70°N counts as about 190 m east–west and
556 m north–south. Next to sea they use one-sided differences, so the coast does not show
up as a cliff. Aspect is the direction the slope faces, in degrees from north.

The BN's `sun_exposure` node has four quadrant states. The north quadrant wraps around 0°,
which breakpoints cannot express, so `aspect_quadrant` turns the aspect into codes
0 = N, 1 = E, 2 = W, 3 = S first.

### 3. Load the BN and wire inputs

```python
bn = geobn.load("avalanche_risk.bif")
bn.set_grid(CRS, RESOLUTION, (WEST, SOUTH, EAST, NORTH))

bn.set_input("slope_angle",  slope)
bn.set_input("sun_exposure", sun_exposure)
bn.set_input("forest_cover", forest_cover)
bn.set_input("recent_snow",  geobn.ConstantSource(RECENT_SNOW_CM))
bn.set_input("temperature",  geobn.ConstantSource(AIR_TEMP_C))
bn.set_input("wind_load",    geobn.ConstantSource(WIND_SPEED_MS))
```

### 4. Read the terrain arrays for the summary

```python
slope_deg = bn.fetch_raw(slope)   # (80, 240) float32, NaN at sea
```

`fetch_raw()` returns the values of any source on the grid. The script uses it for the
console summary and for the slope, exposure and forest layers on the map.

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
