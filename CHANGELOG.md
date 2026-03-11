# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-03-11

### Added
- `GeoBayesianNetwork` — load a `.bif` Bayesian network and wire data sources to evidence nodes.
- `load(path)` factory function for loading BIF models.
- Six built-in data sources: `ArraySource`, `ConstantSource`, `RasterSource`, `URLSource`, `WCSSource`, `PointGridSource`.
- `WCSSource` — generic OGC WCS client (v2.0.1 / v1.1.1 / v1.0.0) with `valid_range` nodata masking.
- `PointGridSource` — sample any `fn(lat, lon) -> float` over an N×N bounding-box grid.
- Pixel-wise BN inference with unique-combination batching (one pgmpy query per unique evidence combo).
- `InferenceResult` with `to_geotiff()`, `to_xarray()`, and `show_map()` (interactive folium map).
- Shannon entropy band per query node in all outputs.
- Real-time optimisation: `bn.freeze(*nodes)` caches static node arrays across `infer()` calls.
- `bn.precompute(query)` pre-solves all evidence combinations into a lookup table; subsequent `infer()` calls run as O(H×W) array indexing.
- Disk caching for `URLSource` and `WCSSource` via `cache_dir` parameter (SHA-256–keyed `.npy` + `.json` pairs).
- Grid alignment via pure numpy + pyproj bilinear interpolation (no rasterio outside source modules).
- `py.typed` marker — full type-hint support for downstream users.
- MkDocs + Material documentation site, auto-deployed to GitHub Pages.
