# geobn

Bayesian network inference over geospatial data.

`geobn` lets you wire data sources — rasters, remote APIs, or plain scalars — directly into a Bayesian network and run pixel-wise inference, producing posterior probability maps and entropy rasters.

---

## Install

```bash
pip install geobn                # core (no rasterio)
pip install "geobn[io]"          # + GeoTIFF export via rasterio
```

Requires Python ≥ 3.13.

---

## Quick start

```python
import geobn
from pathlib import Path

# Load a Bayesian network from a .bif file
bn = geobn.load(Path("fire_risk.bif"))

# Attach data sources to evidence nodes
bn.set_input("slope",    geobn.ArraySource(slope_array,    crs="EPSG:32632", transform=transform))
bn.set_input("rainfall", geobn.ArraySource(rainfall_array, crs="EPSG:32632", transform=transform))

# Define breakpoints for continuous → discrete conversion
bn.set_discretization("slope",    [0, 10, 30, 90],  ["flat", "moderate", "steep"])
bn.set_discretization("rainfall", [0, 25, 75, 200], ["low",  "medium",   "high"])

# Run inference
result = bn.infer(query=["fire_risk"])

probs = result.probabilities["fire_risk"]   # (H, W, 3) — one band per state
ent   = result.entropy("fire_risk")         # (H, W)    — Shannon entropy in bits

# Export
result.to_xarray()          # xarray Dataset  (no rasterio needed)
result.to_geotiff(out_dir)  # multi-band GeoTIFF (requires geobn[io])
```

---

## How it works

```
DataSources  →  align to grid  →  discretize  →  BN inference  →  InferenceResult
```

1. **Load a BN** — `geobn.load("model.bif")` reads a standard `.bif` file via pgmpy.
2. **Attach sources** — each evidence node gets a `DataSource`. All sources are reprojected and resampled to a common grid at inference time (first georeferenced source sets the grid, or call `bn.set_grid()` explicitly).
3. **Discretize** — `set_discretization(node, breakpoints, labels)` bins continuous raster values into the discrete states your BN expects.
4. **Infer** — unique evidence combinations are batched; pgmpy `VariableElimination` runs once per unique combo, not once per pixel.
5. **Export** — `InferenceResult` gives you a numpy array, an xarray Dataset, or a multi-band GeoTIFF (N probability bands + entropy).

---

## Data sources

| Class | Use case |
|---|---|
| `ArraySource(array, crs, transform)` | In-memory numpy array (QGIS, preprocessed data) |
| `RasterSource(path)` | Local GeoTIFF / any rasterio-readable file |
| `URLSource(url)` | Remote Cloud-Optimised GeoTIFF |
| `OpenMeteoSource(variable, date)` | Live weather from [open-meteo.com](https://open-meteo.com/) |
| `ConstantSource(value)` | Broadcast a scalar over the entire grid |

---

## Examples

| Example | Description |
|---|---|
| [`examples/synthetic_fire_risk/`](examples/synthetic_fire_risk/) | Offline example with generated slope + rainfall arrays |
| [`examples/calabria_wildfire/`](examples/calabria_wildfire/) | Real-data example: Copernicus DEM + Open-Meteo weather, Calabria, Italy |

Run either from the repo root:

```bash
uv run python examples/synthetic_fire_risk/run_example.py
uv run python examples/calabria_wildfire/run_example.py
```

---

## Development

```bash
uv pip install -e ".[dev]"
uv run pytest tests/ -v
```
