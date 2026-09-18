# geobn

[![Tests](https://github.com/jensbremnes/geobn/actions/workflows/tests.yml/badge.svg)](https://github.com/jensbremnes/geobn/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

Bayesian network inference over geospatial data.

![geobn demo](docs/assets/demo.gif)

geobn connects geographic data to a Bayesian network and runs inference at every pixel of a map. Inputs can be local rasters, remote services, point APIs or plain numbers. The output is a posterior probability map for each query node, along with an entropy map. Nothing in the library is domain-specific. Typical uses are environmental risk assessment and risk maps for route planning.

Pixels are grouped by their combination of evidence states, so each combination is solved once however many pixels share it. When there are many combinations, geobn computes the whole conditional table in one joint query and looks the pixels up in it. Remote sources can be cached on disk. `bn.precompute()` builds the full lookup table ahead of time, and you can save it with `bn.save_precomputed()` and load it on a machine that doesn't need to run pgmpy.

Documentation: **https://jensbremnes.github.io/geobn**

## Install

```bash
pip install geobn
```

To run the bundled examples, clone the repo:

```bash
git clone https://github.com/jensbremnes/geobn.git
cd geobn
pip install -e ".[dev]"
```

## Data sources

| Class | Use case |
|---|---|
| `ArraySource(array, crs, transform)` | In-memory numpy array |
| `ConstantSource(value)` | Broadcast a scalar over the entire grid |
| `RasterSource(path)` | Local GeoTIFF / any rasterio-readable file |
| `URLSource(url, timeout, cache_dir)` | Remote Cloud-Optimised GeoTIFF |
| `WCSSource(url, layer, valid_range=...)` | Generic OGC WCS endpoint (terrain, bathymetry, …) |
| `PointGridSource(fn, sample_points, delay)` | Sample any `fn(lat, lon) -> float` over the bounding box with user-defined resolution |

## How it works

```
DataSources  →  align to grid  →  discretize  →  BN inference  →  InferenceResult
```

1. Load a network from a `.bif` file with `geobn.load("model.bif")` (read through pgmpy).
2. Attach a `DataSource` to each evidence node. At inference time, all sources are reprojected and resampled onto one grid. By default this is the grid of the georeferenced source with the finest resolution (compared in metres). You can also set it yourself with `bn.set_grid()`.
3. Give each continuous input a set of breakpoints with `set_discretization(node, breakpoints)`, so its values map to the node's discrete states.
4. Run `bn.infer()`. Pixels are grouped by evidence combination and never queried one by one. If the map contains only a few distinct combinations, each gets its own pgmpy `VariableElimination` query. If it contains many, geobn computes P(query | evidence) for all of them in one joint query and indexes into the result. See [How it works](https://jensbremnes.github.io/geobn/concepts/#inference-batching) for the details.
5. Export the `InferenceResult` as numpy arrays, an xarray Dataset, or a multi-band GeoTIFF (one band per state plus entropy).

## Usage

The snippets below are based on the Lyngen Alps avalanche model in [`examples/lyngen_alps/`](examples/lyngen_alps/). They show one of each source type; the example script itself uses only some of them.

### Loading a network

```python
import geobn

bn = geobn.load("avalanche_risk.bif")
bn.set_grid("EPSG:4326", resolution=0.005, extent=(19.8, 69.35, 21.0, 69.75))
```

### Connecting data sources

Each evidence node gets a `DataSource`. Creating a source doesn't fetch anything. Data is loaded when you call `bn.infer()`, or `bn.fetch_raw()` if you want the array yourself.

```python
# WCSSource: fetch data (e.g. terrain) from a WCS server
dtm = geobn.WCSSource(
    url="https://hoydedata.no/arcgis/services/las_dtm_somlos/ImageServer/WCSServer",
    layer="las_dtm",
    version="1.0.0",
    valid_range=(-500, 9000),  # replaces out-of-range sentinel values with NaN
    cache_dir="cache/",
)

# You can also fetch the raw numpy array and process it yourself
dtm_array = bn.fetch_raw(geobn.WCSSource(...))
slope_deg, sun_exposure = my_custom_function(dtm_array)

# ArraySource without a CRS: numpy arrays already on the grid
bn.set_input("slope_angle",  geobn.ArraySource(slope_deg))
bn.set_input("sun_exposure", geobn.ArraySource(sun_exposure))

# RasterSource: local GeoTIFF
bn.set_input("forest_cover", geobn.RasterSource("forest_cover.tif"))

# URLSource: remote Cloud-Optimised GeoTIFF
bn.set_input("recent_snow", geobn.URLSource("https://example.com/recent_snow.tif"))

# PointGridSource: sample any fn(lat, lon) -> float over the bounding box.
# Handy for point weather APIs (MET Norway Frost, Open-Meteo, etc.)
import requests
def fetch_wind_speed(lat, lon):
    r = requests.get(f"https://api.example.com/wind?lat={lat}&lon={lon}")
    return r.json()["wind_speed_ms"]

bn.set_input("wind_load", geobn.PointGridSource(fetch_wind_speed, sample_points=20))

# ConstantSource: one value for the whole grid
bn.set_input("temperature", geobn.ConstantSource(-5.0))   # °C
```

### Discretizing continuous inputs

Breakpoints are bin edges. A node with n states needs n + 1 breakpoints.

```python
bn.set_discretization("slope_angle",  [0, 5, 25, 40, 90])          # degrees
bn.set_discretization("sun_exposure", [-0.5, 0.5, 1.5, 2.5, 3.5])  # N/E/W/S
bn.set_discretization("forest_cover", [-0.5, 0.5, 1.5, 2.5])       # sparse/moderate/dense
bn.set_discretization("wind_load",    [0, 5, 15, 50])              # m/s
bn.set_discretization("recent_snow",  [0, 10, 25, 150])            # cm
bn.set_discretization("temperature",  [-40, -8, -2, 15])           # °C
```

### Running inference

```python
result = bn.infer(query=["avalanche_risk"])
```

The result holds the posterior distribution of each query node at every pixel:

```python
probs = result.probabilities["avalanche_risk"]  # (H, W, n_states), one band per state
ent   = result.entropy("avalanche_risk")         # (H, W), Shannon entropy in bits
p_hi  = result.exceedance("avalanche_risk", "high")  # (H, W), P(risk >= high)

# State names come from the .bif file
for i, state in enumerate(result.state_names["avalanche_risk"]):
    print(f"P({state}) mean: {probs[..., i].mean():.3f}")
```

### Exporting results

```python
result.to_xarray()          # xarray Dataset
result.to_geotiff("out/")   # multi-band GeoTIFF: N probability bands + entropy
result.show_map("out/")     # interactive Leaflet map
```

### Caching remote data to disk

`URLSource` and `WCSSource` take a `cache_dir` argument. Fetched data is saved there as `.npy` files and reused on later runs, including after a restart. The cache key is a SHA-256 hash of the URL and request parameters (bounding box, resolution, layer), so a different grid or source gets a fresh fetch.

```python
dtm = geobn.WCSSource(
    url="https://hoydedata.no/arcgis/services/las_dtm_somlos/ImageServer/WCSServer",
    layer="las_dtm",
    version="1.0.0",
    valid_range=(-500, 9000),
    cache_dir="cache/",
)

snow = geobn.URLSource("https://example.com/recent_snow.tif", cache_dir="cache/")
```

This helps when you are tuning breakpoints or the network structure, since the terrain only has to be downloaded once.

### Repeated inference with changing inputs

If some inputs are static (terrain) and others change between runs (weather), freeze the static ones. They are then fetched and discretized only once:

```python
# Fetched and cached on the first infer() call
bn.freeze("slope_angle", "sun_exposure", "forest_cover")

for wind_ms in [3, 8, 20]:
    bn.set_input("wind_load", geobn.ConstantSource(wind_ms))
    result = bn.infer(query=["avalanche_risk"])
    result.to_geotiff(f"out/wind_{wind_ms}ms/")
```

To go further, `bn.precompute()` solves every combination of evidence states up front, using one joint pgmpy query. Later `infer()` calls then just index into that table.

```python
bn.precompute(query=["avalanche_risk"])
result = bn.infer(query=["avalanche_risk"])  # table lookup, no pgmpy queries
```

You can save the table and load it on another machine, for example a robot. No pgmpy inference runs there:

```python
# Workstation
bn.precompute(query=["avalanche_risk"])
bn.save_precomputed("avalanche_table.npz")

# Target machine
bn.load_precomputed("avalanche_table.npz")
result = bn.infer(query=["avalanche_risk"])
```

## Examples

| Example | Description |
|---|---|
| [`examples/lyngen_alps/`](examples/lyngen_alps/) | Avalanche risk from the Kartverket DTM (WCS) and configurable weather, Lyngen Alps, Norway |
| [`examples/karmsundet/`](examples/karmsundet/) | USV risk from EMODnet bathymetry, AIS traffic density and live MET Norway forecasts, Karmsundet, Norway |

Run from the repo root:

```bash
python examples/lyngen_alps/run_example.py
```

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup.

## Academic foundation

geobn builds on ideas from the author's PhD research. If you use it in academic work, please cite:

> J. E. Bremnes, I. B. Utne, T. R. Krogstad, and A. J. Sørensen,
> "Holistic Risk Modeling and Path Planning for Marine Robotics,"
> *IEEE Journal of Oceanic Engineering*, vol. 50, no. 1, pp. 252–275, 2025.
> DOI: [10.1109/JOE.2024.3432935](https://doi.org/10.1109/JOE.2024.3432935)

## Declaration of AI use

This library was written with the assistance of Claude (Anthropic). All concepts, design decisions, and research ideas originate with the author.
