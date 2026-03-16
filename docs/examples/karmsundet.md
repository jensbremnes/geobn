# Karmsundet — USV Maritime Risk

**Location:** Karmsundet strait, Haugesund, western Norway (59.25°N–59.55°N, 5.15°E–5.55°E)

Karmsundet is a narrow tidal channel between Karmøy island and the mainland — one of Norway's
busiest coastal waterways for commercial shipping, fishing vessels, and recreational traffic.
This example demonstrates real-time maritime risk inference for an unmanned surface vessel (USV)
operating in the area.

## What it demonstrates

- Fetching bathymetry from the **EMODnet** global WCS (no credentials required)
- Sampling **live Met.no forecasts** (wave height, wind, current, fog) via `PointGridSource`
- Using a pre-processed **AIS traffic density** raster from Kystverket open data
- A **3-layer BN** covering grounding risk, collision risk, and navigation difficulty
- **Tier-2 precompute** (`bn.precompute()`) for sub-second inference on 1,215 evidence combinations

## Bayesian network structure

```
water_depth ──────────────────┐
current_speed ────────────────┼──► grounding_risk ─────────┐
                              │                             │
vessel_traffic ───────────────┐                             │
wave_height ──────────────────┼──► collision_risk ──────────┼──► usv_risk
fog_fraction ─────────────────┘                             │
                                                            │
wave_height ──────────────────┐                             │
wind_speed ───────────────────┼──► navigation_difficulty ───┘
current_speed ────────────────┘
```

Six root nodes (evidence inputs), three intermediate nodes, one query node
(`usv_risk`) with states `{low, medium, high}`.

Note that `wave_height` and `current_speed` are parents of two intermediate
nodes each — this is valid in pgmpy and requires no special handling.

## Data sources

| Node | Source | Details |
|------|--------|---------|
| `water_depth` | `WCSSource` (EMODnet Bathymetry) | ~115 m global coverage; cached locally |
| `vessel_traffic` | `RasterSource` (AIS density GeoTIFF) | enc/km²/day; falls back to `ConstantSource(2.0)` |
| `wave_height` | `PointGridSource` (Met.no Oceanforecast) | `sea_surface_wave_height`, 5×5 grid |
| `current_speed` | `PointGridSource` (Met.no Oceanforecast) | `sea_water_speed`, 5×5 grid |
| `wind_speed` | `PointGridSource` (Met.no Locationforecast) | `wind_speed`, 5×5 grid |
| `fog_fraction` | `PointGridSource` (Met.no Locationforecast) | `fog_area_fraction`, 5×5 grid |

## Annotated walkthrough

### 1. Define the study area and grid

```python
WEST, SOUTH, EAST, NORTH = 5.15, 59.25, 5.55, 59.55
CRS = "EPSG:4326"
RESOLUTION = 0.002   # ~200 m at 59°N  →  150 rows × 200 cols

bn = geobn.load("usv_risk.bif")
bn.set_grid(CRS, RESOLUTION, (WEST, SOUTH, EAST, NORTH))
```

### 2. Fetch bathymetry and post-process

```python
raw_depth = bn.fetch_raw(geobn.WCSSource(
    url="https://ows.emodnet-bathymetry.eu/wcs",
    layer="emodnet:mean",
    version="2.0.1",
    valid_range=(-1000.0, 100.0),
    cache_dir=CACHE_DIR,
))

# EMODnet convention: negative = below sea, positive = land
depth = -raw_depth        # positive depth below surface
depth[depth < 0] = np.nan  # land pixels → NaN (no inference)
```

Cached after the first run — subsequent runs load from `examples/karmsundet/cache/`.

### 3. Sample live weather via PointGridSource

```python
def _make_ocean_fn(variable_name):
    def _fn(lat, lon):
        url = f"https://api.met.no/weatherapi/oceanforecast/2.0/compact?lat={lat:.4f}&lon={lon:.4f}"
        req = urllib.request.Request(url, headers={"User-Agent": "geobn/karmsundet"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return float(data["properties"]["timeseries"][0]["data"]["instant"]["details"][variable_name])
    return _fn

bn.set_input("wave_height", geobn.PointGridSource(fn=_make_ocean_fn("sea_surface_wave_height"), sample_points=5))
bn.set_input("current_speed", geobn.PointGridSource(fn=_make_ocean_fn("sea_water_speed"), sample_points=5))
```

Each `PointGridSource` makes 25 API calls (5×5 grid), then `align_to_grid()` bilinearly
resamples the coarse result to the full 150×200 pixel grid.

### 4. Wire all inputs and set discretization

```python
bn.set_input("water_depth",     geobn.ArraySource(depth))
bn.set_input("vessel_traffic",  geobn.RasterSource(ais_path))  # or ConstantSource fallback

bn.set_discretization("water_depth",     [0, 5, 20, 50, 200, 2000])
bn.set_discretization("vessel_traffic",  [0.0, 1.0, 3.0, 1000.0])
bn.set_discretization("wave_height",     [0.0, 0.5, 1.5, 15.0])
bn.set_discretization("wind_speed",      [0.0, 5.0, 12.0, 50.0])
bn.set_discretization("current_speed",   [0.0, 0.3, 1.0, 5.0])
bn.set_discretization("fog_fraction", [0.0, 0.2, 0.6, 1.01])  # 1.01 ensures 1.0 → dense
```

### 5. Precompute and infer

```python
bn.precompute(query=["usv_risk"])   # 1,215 unique combos — sub-second

result = bn.infer(query=["usv_risk"])
```

After `precompute()`, `infer()` uses O(H×W) fancy-index table lookup — zero pgmpy
calls regardless of grid size.

### 6. Scalar risk score and export

```python
probs      = result.probabilities["usv_risk"]   # (H, W, 3): low / medium / high
risk_score = (probs * np.array([10.0, 50.0, 90.0])).sum(axis=-1)

result.to_geotiff(OUT_DIR)   # 4-band GeoTIFF: P(low), P(medium), P(high), entropy
result.show_map(OUT_DIR, filename="usv_risk_map.html",
                extra_layers={"Risk score (10–90)": risk_score, "Water depth (m)": depth})
```

## Key outputs

- **`output/usv_risk_map.html`** — interactive Leaflet map with USV risk probability,
  entropy, risk score, and depth overlays
- **`output/usv_risk.tif`** — 4-band GeoTIFF:
  Band 1 P(low), Band 2 P(medium), Band 3 P(high), Band 4 entropy
- **`output/risk_score.tif`** — scalar risk score in the range 10–90

## AIS traffic density

The script falls back to `ConstantSource(2.0)` when no AIS raster is present.
To generate a real raster from Kystverket open data:

1. Download historical AIS CSV from [kystdatahuset.no](https://kystdatahuset.no)
   (select the Karmsundet bounding box and a date range, e.g. 30 days)
2. Run:

```bash
uv run python examples/karmsundet/create_ais_density.py aisdata.csv 30
```

This produces `data/ais_density_karmsundet.tif` — a float32 GeoTIFF with traffic
density in encounters/km²/day on the same 150×200 grid.

## How to run

```bash
uv run python examples/karmsundet/run_example.py
```

The bathymetry is cached on first run. The Met.no forecasts are fetched live on
every run (~100 API calls total, taking roughly 5–10 seconds).
