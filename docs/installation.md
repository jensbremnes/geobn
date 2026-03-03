# Installation

geobn requires **Python ≥ 3.13** and is managed with [uv](https://docs.astral.sh/uv/).

## Extras

geobn uses optional dependency groups so you only install what you need:

| Extra | What it adds | Install command |
|-------|-------------|-----------------|
| *(core)* | Inference engine, all built-in sources, reprojection | `pip install geobn` |
| `[io]` | GeoTIFF read/write via `rasterio` | `pip install "geobn[io]"` |
| `[viz]` | Interactive maps via `folium`, plots via `matplotlib` | `pip install "geobn[viz]"` |
| `[ocean]` | Copernicus Marine SDK + STAC/xarray access | `pip install "geobn[ocean]"` |
| `[full]` | `[io]` + `[viz]` + `xarray` — recommended for analysis | `pip install "geobn[full]"` |

### When to use each extra

**`[io]`** — needed if you call `result.to_geotiff()` or use `RasterSource` /
`URLSource` to load local or remote GeoTIFFs.

**`[viz]`** — needed if you call `result.show_map()`.

**`[ocean]`** — needed if you use `CopernicusMarineSource` (CMEMS) or `HubOceanSource`
(STAC catalog). These sources require separate credentials.

**`[full]`** — recommended for notebooks and analysis workflows; adds `xarray` so
`result.to_xarray()` works, and includes `[io]` and `[viz]`.

## Recommended install

```bash
pip install "geobn[full]"
```

Or with uv (faster):

```bash
uv pip install "geobn[full]"
```

## Development install

```bash
git clone https://github.com/jensebr/geobn.git
cd geobn
uv pip install -e ".[dev]"
```

The `dev` extra pulls in `[full]` plus `pytest`.

## Docs install

To build the documentation locally:

```bash
uv pip install ".[docs]"
mkdocs serve   # browse at http://127.0.0.1:8000
```
