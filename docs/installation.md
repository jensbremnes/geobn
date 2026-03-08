# Installation

geobn requires **Python ≥ 3.13** and is managed with [uv](https://docs.astral.sh/uv/).

## Install

```bash
pip install geobn
```

Or with uv (faster):

```bash
uv pip install geobn
```

All features — GeoTIFF I/O, interactive maps, xarray output, and all built-in data
sources — are included in the standard install.

## Development install

```bash
git clone https://github.com/jensebr/geobn.git
cd geobn
uv pip install -e ".[dev]"
```

The `dev` extra pulls in pytest.

## Docs install

To build the documentation locally:

```bash
uv pip install -e ".[docs]"
mkdocs serve   # browse at http://127.0.0.1:8000
```
