# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `geobn.load()` now dispatches on the file extension and reads `.bif`, `.xmlbif`, `.net` (Hugin), `.xdsl` (GeNIe) and `.uai`, in addition to `.bif`. All readers already ship with pgmpy, so there is no new dependency. A bare `.xml` file is resolved from its root element: `<BIF>` → XMLBIF, `<smile>` → GeNIe XDSL, `<ANALYSISNOTEBOOK>` → Microsoft XBN; any other root element raises `ValueError`. UAI files store no variable or state names, so their nodes arrive as `var_0`, `var_1`, … and their states as integers — loading one emits a `UserWarning` saying so. Netica `.dne` is not supported (pgmpy has no reader); the error message says to export to `.net` or `.bif` instead.
- Documented that `GeoBayesianNetwork(model)` accepts a pgmpy `DiscreteBayesianNetwork` directly, for models built in code, fitted with a pgmpy estimator, or read with a reader `load()` does not dispatch on.
- Summary maps on `InferenceResult`: `expected_value(node, values)`, `std(node, values)`, `mode(node)`, `mode_probability(node)`, `exceedance(node, state)` (P ≥ state), `ignorance(node, threshold)` and `quantile(node, q)`. Each returns a `(H, W)` float32 array, NaN for NoData. `values` is a list in state order or a dict keyed by state name.
- `to_geotiff(..., layers={name: array})` writes extra `(H, W)` layers as single-band `{name}.tif` files; `to_xarray(layers=...)` adds them as `(y, x)` variables.
- `save_precomputed(path)` and `load_precomputed(path)` methods on `GeoBayesianNetwork` — serialize the precomputed lookup table to a portable `.npz` file for offline→runtime deployment.
- `query_point(evidence)` and `query_batch(evidence)` on `GeoBayesianNetwork`: posteriors for single points or batches of points, looked up directly in the precomputed table without a grid. Evidence may be numbers or state names; NaN gives NaN probabilities.
- `set_discretization(..., out_of_range="clip" | "nan")`: choose whether values outside `[breakpoints[0], breakpoints[-1]]` are clipped to the first/last state (default, unchanged behaviour) or treated as NoData.
- `save_precomputed` stores a format version, the geobn version, a hash of the model (structure, state names, CPD values), the BN state names and each input's discretization. `load_precomputed` checks all of these and raises `ValueError` if the table was built from a different model or with a different discretization.
- `WCSSource(axis_labels=(lon_label, lat_label))`: set the axis names used in WCS 2.0 `SUBSET` parameters, for servers that don't use the default `Long`/`Lat` (e.g. `lon`/`lat` or `x`/`y`).
- `load_precomputed` restores the saved discretizations for inputs that have none set, so a runtime machine only needs to register the inputs.
- `set_input(node, source)` accepts any node in the BN, not only root nodes. A dataset that measures an intermediate concept can be used in place of the layers that feed it, and querying an ancestor of an observed node gives a posterior over that ancestor. On the bundled Lyngen model, observing `terrain_factor="high"` gives P(`slope_angle`) = 0.006 / 0.039 / 0.372 / 0.583 over `flat, gentle, steep, extreme`, against a uniform 0.25 prior. Inputs on nodes with parents are not independent of one another, so they can contradict the model: observing a node together with its parent in a combination the CPD gives probability zero yields NaN and a `UserWarning`, as any other impossible evidence does.
- A node cannot be both an input and a query node. `infer()`, `precompute()` and `load_precomputed()` raise `ValueError` naming the overlapping nodes, before any data is fetched.
- `valid_range=(lo, hi)` on every data source, not only `WCSSource`. Values below `lo` or above `hi` are replaced with NaN, for data that encodes missing values as an extreme number without declaring it as nodata. Both bounds are inclusive and either may be `None` to leave that side unbounded, e.g. `valid_range=(0.0, None)`. `DataSource.fetch()` applies the mask around the `_fetch()` each source implements, so a source written outside geobn gets the same behaviour by taking the argument and passing it to `super().__init__()`.
- `cache_ttl` on `URLSource`, `WCSSource` and `PointGridSource`: the maximum age of a disk-cache entry, as a `timedelta` or a number of seconds. An older entry is fetched again; the default `None` never expires, which suits terrain and bathymetry. `cache_ttl` is not part of the cache key, so changing it re-reads the same entry rather than orphaning it. If an entry has expired and the fetch then fails, that entry is returned with a `UserWarning` naming its age, so an offline run still produces a map.
- `PointGridSource(name=..., cache_dir=..., cache_ttl=...)` caches its sampled lattice on disk; on a hit the callable is not called at all, which for the default 5×5 lattice is 25 API calls saved. `name` identifies the cache entry, because a callable cannot identify itself — two sources built by the same factory are indistinguishable and would share one entry. Passing `cache_dir` without `name` raises `ValueError`.

### Changed
- `geobn.load()` now validates the parsed model with pgmpy's `check_model()` and raises `ValueError` naming the file. A malformed or truncated file, or a CPT whose columns do not sum to 1, now fails at load time instead of deep inside `infer()`. An unreadable file and an unknown extension also raise `ValueError`, and a missing file raises `FileNotFoundError`, instead of surfacing a raw pgmpy or parser error.

### Fixed
- `geobn.load()` decodes `.bif`, `.net` and `.uai` files as UTF-8 (falling back to the platform default encoding), instead of letting pgmpy open them with the platform default. A UTF-8 model file with non-ASCII characters — including the bundled `examples/lyngen_alps/avalanche_risk.bif`, whose header comment draws the network with box-drawing characters — raised `UnicodeDecodeError` on Windows. XML-based formats were unaffected, as the XML parser honours the file's own encoding declaration.
- Evidence with probability zero (e.g. an input observed in a state with prior 0) now gives NaN for every query node on every inference path (per-combination loop, joint table, `precompute()` and its fallback). Previously the per-combination loop could return probabilities for query nodes that pgmpy pruned away from the contradiction, and for children of a root observed in a zero-prior state. `infer()` and `query_batch()`/`query_point()` now also emit a `UserWarning` giving the number of affected pixels/points and the observed states with prior 0 (e.g. `Weather='storm'`).
- `to_geotiff` now writes the band descriptions (state labels and `"entropy"`) that the docs already promised.
- `WCSSource` disk cache: the cache key now includes `extra_subsets`, `format` and `axis_labels`. Previously, e.g. two time steps of the same coverage (different `extra_subsets`) shared one cache entry, so the second returned the first one's data. Cache entries for requests that use none of these options stay valid.
- `valid_range` was accepted without validation, so `(9000, -500)` silently masked everything, and a bare float or a NaN bound failed later with an unrelated error. It is now checked in the constructor and raises `ValueError` naming the value.
- The `valid_range` mask no longer writes into the fetched array in place, which would have modified the array a caller handed to `ArraySource`.
- `GridSpec.extent_wgs84()` now transforms 21 points along each grid edge instead of only the 4 corners, so `WCSSource` and `PointGridSource` no longer request a box that is too small for large projected grids (edges curve in lon/lat; up to ~12 km was missed on a 1000 km grid). A grid that contains a pole now gives a box reaching that pole and spanning all longitudes.
- Lyngen example: slope no longer treats sea/nodata as 0 m elevation, which created fake cliffs along the coast (a flat coastal plateau got ~38°). Differences next to nodata are now one-sided. Aspect classes now use the direction the slope faces; north/south and east/west were swapped.
- Discretization labels given in a different order from the BN's states (e.g. `["high", "medium", "low"]` for a node defined as `low, medium, high`) gave wrong posteriors from the precomputed table (`infer` after `precompute`, `query_point`, `query_batch`). The pgmpy path was correct. Indices are now mapped to the BN's state order in every path.
- `RasterSource`, `URLSource` and `WCSSource` now convert the GeoTIFF's declared nodata value (and internal masks) to NaN. Previously nodata sentinels such as −9999 were discretized as real values, producing confident but wrong posteriors.
- An `ArraySource` without CRS whose shape does not match the grid now raises `ValueError`. Previously it was silently broadcast from its first value.
- `docs/concepts.md` wrongly said that values outside the breakpoint range become NaN; they are clipped unless `out_of_range="nan"` is set.
- Both example scripts write their output as UTF-8 instead of the platform default. They print `→`, `─` and `≤`, which the Windows console encoding (cp1252) cannot represent, so `examples/karmsundet/run_example.py` died with `UnicodeEncodeError` partway through and `examples/lyngen_alps/run_example.py` would have done the same.
- Automatic reference-grid selection now compares pixel sizes in metres instead of CRS units. Previously a 0.001° WGS84 source (~80 m) was chosen over a 10 m UTM source, downsampling the finer data.

### Changed
- Multi-node `infer()` is much faster for large query sets (#18). pgmpy builds the full joint over all query nodes even with `joint=False`, so one call for 19 query nodes (192M cells) took ~10 s per evidence combination. Query nodes are now requested in one call only when their joint table has at most 20,000 cells, and one by one otherwise (32 ms in the same case). The `precompute()` fallback loop uses the same strategy.
- `load_precomputed` no longer requires inputs to be registered in the same order as when the table was saved; the table axes are reordered by node name. Tables saved by older versions still load, with a `UserWarning`.
- Disk cache keys include a cache-format version. Existing `cache_dir` entries (which may contain unmasked nodata) are ignored and refetched once; old files can be deleted manually.
- `WCSSource` caches the array the server sent and applies `valid_range` on the way out, so the range is no longer part of the cache key. Two sources that differ only in `valid_range` now share one cache entry, and changing the range re-masks the cached data instead of requesting the coverage again.
- Data sources implement `_fetch(grid)` instead of `fetch(grid)`; `DataSource.fetch()` is provided by the base class and applies `valid_range` around it.
- The disk cache is handled by `DataSource.fetch()` rather than by each source: it reads the cache, calls `_fetch()` on a miss, writes the result back, then applies `valid_range`. A source becomes cacheable by overriding `_cache_key(grid)`, which returns a JSON-serializable dict identifying the request or `None` when the source is not cached. `WCSSource._cache_key` takes the grid instead of six positional floats, and returns the same key as before, so existing cache entries stay valid.
- Cache sidecars record `fetched_at`, the unix time the entry was written. Entries written without it are dated by the array file's modification time, so caches from earlier versions keep working without a refetch.

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
