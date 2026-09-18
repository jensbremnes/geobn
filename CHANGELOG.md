# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Summary maps on `InferenceResult`: `expected_value(node, values)`, `std(node, values)`, `mode(node)`, `mode_probability(node)`, `exceedance(node, state)` (P ≥ state), `ignorance(node, threshold)` and `quantile(node, q)`. Each returns a `(H, W)` float32 array, NaN for NoData. `values` is a list in state order or a dict keyed by state name.
- `to_geotiff(..., layers={name: array})` writes extra `(H, W)` layers as single-band `{name}.tif` files; `to_xarray(layers=...)` adds them as `(y, x)` variables.
- `save_precomputed(path)` and `load_precomputed(path)` methods on `GeoBayesianNetwork` — serialize the precomputed lookup table to a portable `.npz` file for offline→runtime deployment.
- `query_point(evidence)` and `query_batch(evidence)` on `GeoBayesianNetwork`: posteriors for single points or batches of points, looked up directly in the precomputed table without a grid. Evidence may be numbers or state names; NaN gives NaN probabilities.
- `set_discretization(..., out_of_range="clip" | "nan")`: choose whether values outside `[breakpoints[0], breakpoints[-1]]` are clipped to the first/last state (default, unchanged behaviour) or treated as NoData.
- `save_precomputed` stores a format version, the geobn version, a hash of the model (structure, state names, CPD values), the BN state names and each input's discretization. `load_precomputed` checks all of these and raises `ValueError` if the table was built from a different model or with a different discretization.
- `WCSSource(axis_labels=(lon_label, lat_label))`: set the axis names used in WCS 2.0 `SUBSET` parameters, for servers that don't use the default `Long`/`Lat` (e.g. `lon`/`lat` or `x`/`y`).
- `load_precomputed` restores the saved discretizations for inputs that have none set, so a runtime machine only needs to register the inputs.

### Fixed
- `to_geotiff` now writes the band descriptions (state labels and `"entropy"`) that the docs already promised.
- `WCSSource` disk cache: the cache key now includes `extra_subsets`, `format`, `valid_range` and `axis_labels`. Previously, e.g. two time steps of the same coverage (different `extra_subsets`) shared one cache entry, so the second returned the first one's data. Cache entries for requests that use none of these options stay valid.
- `GridSpec.extent_wgs84()` now transforms 21 points along each grid edge instead of only the 4 corners, so `WCSSource` and `PointGridSource` no longer request a box that is too small for large projected grids (edges curve in lon/lat; up to ~12 km was missed on a 1000 km grid). A grid that contains a pole now gives a box reaching that pole and spanning all longitudes.
- Lyngen example: slope no longer treats sea/nodata as 0 m elevation, which created fake cliffs along the coast (a flat coastal plateau got ~38°). Differences next to nodata are now one-sided. Aspect classes now use the direction the slope faces; north/south and east/west were swapped.
- Discretization labels given in a different order from the BN's states (e.g. `["high", "medium", "low"]` for a node defined as `low, medium, high`) gave wrong posteriors from the precomputed table (`infer` after `precompute`, `query_point`, `query_batch`). The pgmpy path was correct. Indices are now mapped to the BN's state order in every path.
- `RasterSource`, `URLSource` and `WCSSource` now convert the GeoTIFF's declared nodata value (and internal masks) to NaN. Previously nodata sentinels such as −9999 were discretized as real values, producing confident but wrong posteriors.
- An `ArraySource` without CRS whose shape does not match the grid now raises `ValueError`. Previously it was silently broadcast from its first value.
- `docs/concepts.md` wrongly said that values outside the breakpoint range become NaN; they are clipped unless `out_of_range="nan"` is set.
- Automatic reference-grid selection now compares pixel sizes in metres instead of CRS units. Previously a 0.001° WGS84 source (~80 m) was chosen over a 10 m UTM source, downsampling the finer data.

### Changed
- `load_precomputed` no longer requires inputs to be registered in the same order as when the table was saved; the table axes are reordered by node name. Tables saved by older versions still load, with a `UserWarning`.
- Disk cache keys include a cache-format version. Existing `cache_dir` entries (which may contain unmasked nodata) are ignored and refetched once; old files can be deleted manually.

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
