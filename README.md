# geobn

[![Tests](https://github.com/jensebr/geobn/actions/workflows/tests.yml/badge.svg)](https://github.com/jensebr/geobn/actions/workflows/tests.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Bayesian network inference over geospatial data.

![geobn demo](docs/assets/demo.gif)

> **Under development** — the API is functional and tested, but may change before a stable 1.0 release.

`geobn` lets you turn heterogeneous data sources (offline and real-time) into insight over geographical areas by using techniques in probabilistic AI. The library is domain-agnostic, and may be used for, e.g., environmental risk assessment and risk‑informed route planning.

This is achieved by wiring different data sources — rasters, remote APIs, or plain scalars — directly into a Bayesian network, and run pixel-wise inference, producing posterior probability maps and entropy rasters. Under the hood it uses disk caching of remote data and groups pixels by unique evidence combinations, so each inference query is solved once per combination instead of once per pixel, keeping computations of large areas computationally tractable.

Full docs (API reference, concepts, examples) are hosted at:
**https://jensebr.github.io/geobn**

---

## Install

> **PyPI release coming soon.** Until then, install directly from source (Python ≥ 3.13 required):

```bash
uv pip install git+https://github.com/jensebr/geobn.git
```

To also run the bundled examples, clone the repo instead:

```bash
git clone https://github.com/jensebr/geobn.git
cd geobn
uv pip install -e ".[dev]"
```

---

## How it works

```
DataSources  →  align to grid  →  discretize  →  BN inference  →  InferenceResult
```

1. **Load a BN** — `geobn.load("model.bif")` reads a standard `.bif` file via pgmpy.
2. **Attach sources** — each evidence node gets a `DataSource`. All sources are reprojected and resampled to a common grid at inference time (the finest-resolution georeferenced source sets the grid automatically, or call `bn.set_grid()` explicitly).
3. **Discretize** — `set_discretization(node, breakpoints)` bins continuous values into the discrete states your BN expects.
4. **Infer** — unique evidence combinations are batched; pgmpy `VariableElimination` runs once per unique combo, not once per pixel.
5. **Export** — `InferenceResult` gives you a numpy array, an xarray Dataset, or a multi-band GeoTIFF (N probability bands + entropy).

---

## Usage

### Loading a network

```python
import geobn

bn = geobn.load("fire_risk.bif")
```

### Connecting data sources

Attach a `DataSource` to each evidence node. Sources can be local files, remote services, or plain scalars — they are all reprojected and aligned to a common grid at inference time.

```python
# Local GeoTIFF
bn.set_input("slope",      geobn.RasterSource("slope.tif"))

# Remote terrain model via OGC WCS (cached to disk after first fetch)
bn.set_input("elevation",  geobn.WCSSource(
    url="https://example.com/wcs",
    layer="dtm",
    version="1.0.0",
    valid_range=(-500, 9000),  # replaces out-of-range sentinels with NaN
    cache_dir="cache/",
))

# Any web API that returns a scalar per (lat, lon) point
import requests
def fetch_ndvi(lat, lon):
    r = requests.get(f"https://api.example.com/ndvi?lat={lat}&lon={lon}")
    return r.json()["ndvi"]

bn.set_input("vegetation", geobn.PointGridSource(fetch_ndvi, sample_points=20))

# Broadcast a single scalar over the entire grid — useful for weather scenarios
bn.set_input("recent_rain", geobn.ConstantSource(15.0))  # 15 mm
```

### Discretizing continuous inputs

Breakpoints map continuous raster values into the discrete states your BN expects. The number of intervals must match the number of states for that node.

```python
bn.set_discretization("slope",       [0, 10, 30, 90])     # flat / moderate / steep
bn.set_discretization("elevation",   [0, 500, 1500, 4000]) # low / mid / high
bn.set_discretization("vegetation",  [0.0, 0.3, 0.6, 1.0])
bn.set_discretization("recent_rain", [0, 5, 20, 200])
```

### Running inference

```python
result = bn.infer(query=["fire_risk"])
```

`infer()` returns an `InferenceResult` with a posterior probability array and entropy map for each queried node.

```python
probs = result.probabilities["fire_risk"]  # (H, W, n_states) — one band per state
ent   = result.entropy("fire_risk")        # (H, W) — Shannon entropy in bits

# State names come directly from the .bif file
for i, state in enumerate(result.state_names["fire_risk"]):
    print(f"P({state}) mean: {probs[..., i].mean():.3f}")
```

### Exporting results

```python
result.to_xarray()          # xarray Dataset — integrates with existing geospatial workflows
result.to_geotiff("out/")   # multi-band GeoTIFF: N probability bands + entropy
result.show_map("out/")     # interactive Leaflet map in the browser
```

### Repeated inference with changing inputs

When static inputs (e.g. terrain) are mixed with inputs that change between calls (e.g. weather), freeze the static nodes so their arrays are fetched and discretized only once:

```python
bn.freeze("slope", "elevation", "vegetation")  # fetched and cached on first infer()

# Explore different weather scenarios without re-fetching terrain
for rain_mm in [5, 15, 40]:
    bn.set_input("recent_rain", geobn.ConstantSource(rain_mm))
    result = bn.infer(query=["fire_risk"])
    result.to_geotiff(f"out/rain_{rain_mm}mm/")
```

For maximum throughput, pre-run all evidence combinations once and reduce subsequent calls to a numpy index lookup:

```python
bn.precompute(query=["fire_risk"])  # one-time cost: runs all state combinations
result = bn.infer(query=["fire_risk"])  # O(H×W) array indexing — no pgmpy at runtime
```

---

## Data sources

| Class | Use case |
|---|---|
| `ArraySource(array, crs, transform)` | In-memory numpy array |
| `ConstantSource(value)` | Broadcast a scalar over the entire grid |
| `RasterSource(path)` | Local GeoTIFF / any rasterio-readable file |
| `URLSource(url)` | Remote Cloud-Optimised GeoTIFF |
| `WCSSource(url, layer, valid_range=...)` | Generic OGC WCS endpoint (terrain, bathymetry, …) |
| `PointGridSource(fn)` | Sample any `fn(lat, lon) -> float` over the bounding box |

---

## Examples

| Example | Description |
|---|---|
| [`examples/lyngen_alps/`](examples/lyngen_alps/) | Avalanche risk: Kartverket DTM via WCSSource + configurable weather, Lyngen Alps, Norway |

Run from the repo root:

```bash
uv run python examples/lyngen_alps/run_example.py
```

---

## Academic foundation

`geobn` is a software realisation of ideas developed during the author's PhD research. If you use this library in academic work, please consider citing the following paper:

> J. E. Bremnes, I. B. Utne, T. R. Krogstad, and A. J. Sørensen,
> "Holistic Risk Modeling and Path Planning for Marine Robotics,"
> *IEEE Journal of Oceanic Engineering*, vol. 50, no. 1, pp. 252–275, 2025.
> DOI: [10.1109/JOE.2024.3432935](https://doi.org/10.1109/JOE.2024.3432935)

---

## Declaration of AI use

This library was written with the assistance of Claude (Anthropic). All concepts, design decisions, and research ideas originate with the author.
