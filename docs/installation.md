# Installation

geobn requires Python 3.11 or newer.

## Install

```bash
pip install geobn
```

The standard install includes everything: GeoTIFF I/O, interactive maps, xarray
output and all built-in data sources.

## Development install

```bash
git clone https://github.com/jensbremnes/geobn.git
cd geobn
pip install -e ".[dev]"
```

The `dev` extra adds pytest and pytest-cov.

## Building the docs

```bash
pip install -e ".[docs]"
mkdocs serve   # browse at http://127.0.0.1:8000
```
