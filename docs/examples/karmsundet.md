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
| `vessel_traffic` | `MosaicSource` (AIS density GeoTIFF, then a prior) | enc/km²/day, backed by `ConstantSource(2.0)` |
| `wave_height` | `PointGridSource` (Met.no Oceanforecast) | `sea_surface_wave_height`, 5×5 grid; cached |
| `current_speed` | `PointGridSource` (Met.no Oceanforecast) | `sea_water_speed`, 5×5 grid; cached |
| `wind_speed` | `PointGridSource` (Met.no Locationforecast) | `wind_speed`, 5×5 grid; cached |
| `fog_fraction` | `PointGridSource` (Met.no Locationforecast) | `fog_area_fraction`, 5×5 grid; cached |

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

for node, variable in [("wave_height", "sea_surface_wave_height"),
                       ("current_speed", "sea_water_speed")]:
    bn.set_input(node, geobn.PointGridSource(
        fn=_make_ocean_fn(variable),
        sample_points=5,
        name=node,
        cache_dir=CACHE_DIR,
    ))
```

Each `PointGridSource` makes 25 API calls (5×5 grid), then `align_to_grid()` bilinearly
resamples the coarse result to the full 150×200 pixel grid. The lattice is sampled once and
then kept, because this example is about the output rather than the weather being current;
pass `cache_ttl=timedelta(hours=6)` to re-sample after a given age. `name` is what the cache
entry is keyed on, since the four closures built by these factories are otherwise
indistinguishable.

### 4. Wire all inputs and set discretization

```python
bn.set_input("water_depth", geobn.ArraySource(depth))

# AIS density where the raster has data, a medium-traffic prior everywhere else.
ais_source = geobn.MosaicSource(
    [geobn.RasterSource(ais_path), geobn.ConstantSource(2.0)],
    names=["ais_density", "medium_traffic_prior"],
    on_error="skip",
)
bn.set_input("vessel_traffic", ais_source)

# The provenance layer says which pixels came from which source.
ais_raw, ais_provenance = bn.fetch_raw(ais_source, return_provenance=True)

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
                extra_layers={"Risk score (10–90)": risk_score,
                              "Water depth (m)": depth,
                              "AIS traffic density": ais_raw,
                              "AIS source index": ais_provenance.astype(np.float32)})
```

## Key outputs

- **`output/usv_risk_map.html`** — interactive Leaflet map with USV risk probability,
  entropy, risk score, depth, AIS traffic density and the AIS source index overlays
- **`output/usv_risk.tif`** — 4-band GeoTIFF:
  Band 1 P(low), Band 2 P(medium), Band 3 P(high), Band 4 entropy
- **`output/risk_score.tif`** — scalar risk score in the range 10–90

## AIS traffic density

`ConstantSource(2.0)` stands in for the AIS raster wherever it has no data, including
when the file is absent altogether. `on_error="skip"` is what allows a missing file to
count as no data; the run then prints which share of the grid came from each source.
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

The first run fetches the bathymetry and makes ~100 Met.no calls (roughly 5–10 seconds).
Everything is cached and kept, so later runs make no requests at all. Delete
`examples/karmsundet/cache/` to fetch current forecasts, or pass `cache_ttl` to the
`PointGridSource`s to have them re-sample on their own.
