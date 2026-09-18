# How it works

## Pipeline overview

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌──────────────────┐
│ DataSource   │────►│ align_to_grid()  │────►│ discretize() │────►│ BN inference     │
│ (per node)   │     │ reproject + resample   │ continuous → │     │ VariableElim.    │
└──────────────┘     │ to reference grid│     │ BN states    │     │ batched queries  │
                     └──────────────────┘     └──────────────┘     └──────────┬───────┘
                                                                               │
                                                                    ┌──────────▼───────┐
                                                                    │ InferenceResult  │
                                                                    │ (H, W, n_states) │
                                                                    │ + entropy map    │
                                                                    └──────────────────┘
```

## Loading a Bayesian network

geobn reads `.bif` files (Bayesian Interchange Format), the standard format used by
GeNIe/Netica/bnlearn. Use [`geobn.load()`][geobn.load]:

```python
bn = geobn.load("my_model.bif")
```

## Attaching data sources

Each root node (node with no parents) in the BN corresponds to an evidence variable.
Attach a [`DataSource`][geobn.sources.DataSource] to each one:

```python
bn.set_input("slope_angle", geobn.WCSSource(url="https://example.com/wcs", layer="dtm"))
bn.set_input("recent_snow", geobn.ConstantSource(30.0))
```

At inference time, geobn inspects all registered georeferenced sources and selects the
one with the **finest resolution** as the **reference grid** (CRS, resolution, extent).
Pixel sizes are compared in metres on the ground (measured at each grid's centre), so a
10 m UTM raster correctly beats a 0.001° WGS84 raster even though 0.001 < 10 in CRS units. All other sources are reprojected to this grid
automatically. To override this behaviour, call `bn.set_grid(crs, resolution, extent)`
explicitly before running inference.

## GridSpec and alignment

The reference grid is described by a `GridSpec(crs, transform, shape)`. Grid alignment
uses pure numpy + pyproj bilinear interpolation — no rasterio dependency.

`ConstantSource` is a special case: it returns a 1×1 sentinel array with `crs=None`,
which `align_to_grid()` recognises and broadcasts to the full grid shape.

## Discretization

Bayesian networks operate on discrete states. Every continuous evidence source must be
mapped to BN state names via `set_discretization()`:

```python
bn.set_discretization(
    "slope_angle",
    breakpoints=[0, 25, 40, 90],   # bin edges
    labels=["gentle", "steep", "extreme"],  # must match BN state names exactly
)
```

`breakpoints` must have `len(labels) + 1` values. The interior values are the bin
edges; a value exactly on an edge goes to the upper bin.

The first and last breakpoints define the valid range `[breakpoints[0], breakpoints[-1]]`
(inclusive). By default (`out_of_range="clip"`), values outside it are assigned to the
first or last state, so a slope of 95° counts as `"extreme"`. Pass
`out_of_range="nan"` to treat them as NoData instead:

```python
bn.set_discretization(
    "slope_angle",
    breakpoints=[0, 25, 40, 90],
    labels=["gentle", "steep", "extreme"],
    out_of_range="nan",   # values < 0 or > 90 → NaN output
)
```

This is useful when out-of-range values mean bad data (sensor errors, fill values)
rather than extreme conditions.

## NaN / NoData propagation

NaN values propagate strictly: if **any** input pixel is NaN, that pixel is excluded
from inference and all output bands for that pixel are NaN.

This means:
- Pixels outside WCS coverage → NaN inputs → NaN outputs
- Sea pixels in a land DEM → NaN depth → NaN output
- Invalid sensor readings → NaN evidence → NaN posteriors
- Values outside the breakpoint range with `out_of_range="nan"` → NaN posteriors
- File nodata → NaN: `RasterSource`, `URLSource` and `WCSSource` automatically convert
  pixels matching the GeoTIFF's declared nodata value (or its internal mask) to NaN

An `ArraySource` without `crs`/`transform` must either match the grid shape exactly
(pre-aligned) or be a single value; any other shape raises `ValueError` rather than
being silently broadcast.

## Inference batching

Running one pgmpy `VariableElimination.query()` per pixel is prohibitively slow for
large rasters, so geobn never queries per pixel. Instead, inference proceeds in two
steps: first **measure** how much distinct evidence the map actually contains, then
**choose** the cheapest strategy for that amount.

### Step 1 — group pixels by observed evidence combination

After discretization, every pixel is described by one state index per evidence node
— e.g. `(slope=steep, rainfall=high)`. geobn groups all valid pixels by these
combinations and counts how many **distinct combinations actually occur on the map**.

This count is the key quantity, and it is usually far smaller than the number of
*theoretically possible* combinations:

- It can never exceed the number of valid pixels.
- Geography is spatially correlated — neighbouring pixels tend to share the same
  slope class, forest class, etc.
- Scalar inputs (`ConstantSource`, live weather values) contribute exactly one state
  each, collapsing the combination count further.

A network with millions of *possible* combinations may still produce only a few
dozen *observed* ones, and geobn's strategy choice is driven entirely by the
observed count — the size of the theoretical state space costs nothing by itself.

### Step 2 — choose a strategy

**Few observed combinations (≤ 200): per-combination queries.** pgmpy
`VariableElimination` queries run per observed combination. For a 500×500 grid with
3 evidence nodes (3 states each) there are at most 27 combinations regardless of grid
size — a handful of targeted queries is the cheapest possible approach.

Within one combination, how the query nodes are requested depends on the size of
their joint table (the product of their state counts). pgmpy computes the full joint
over all requested nodes before splitting it into per-node marginals, so:

- **Small joint (≤ 20,000 cells):** all query nodes in one pgmpy call, which shares
  the elimination work (up to ~3× faster than separate calls).
- **Large joint:** one pgmpy call per query node. The joint grows multiplicatively
  with every query node — 19 query nodes with 2–5 states each is 192 million cells —
  while separate calls grow linearly (10 s vs 32 ms per combination in that case).

**Many observed combinations: one joint query.** Asking thousands of nearly
identical questions repeats the same internal propagation work each time. Instead,
geobn asks pgmpy a single bigger question: the **joint** distribution
P(query, e₁, …, eₖ) with no evidence at all. Normalising that array along the query
axis converts it into the full conditional table P(query | e₁, …, eₖ) — the answer
for *every* combination at once:

```
P(query | e₁, …, eₖ) = P(query, e₁, …, eₖ) / P(e₁, …, eₖ)
```

Per-pixel results are then read from the table by numpy fancy indexing. This keeps
networks with many evidence nodes tractable: 10 nodes × 3 states (59,049
combinations) resolves in well under a second, where the per-combination loop would
take minutes.

**Safety valve:** the joint table's size is the product of all evidence state counts
× the query state count. If that exceeds an in-memory bound (~80 MB), the joint
strategy is skipped and the per-combination loop is used — slower, but it only ever
pays for combinations that actually occur.

### Summary of regimes

| Observed combos | Table fits in memory? | Strategy | Cost |
|---|---|---|---|
| ≤ 200 | (irrelevant) | per-combination queries | milliseconds |
| many | yes | single joint query + table lookup | ~constant, sub-second |
| many | no | per-combination queries | proportional to observed combos |

In all cases the per-combination results are scattered back to the original pixel
positions.

**Impossible evidence.** When an evidence combination has probability zero (e.g. an
input observed in a state whose prior is 0), the posterior is undefined and *every*
query node gets NaN, on every path. pgmpy on its own would still return numbers for
query nodes it can prune away from the contradiction, so geobn checks P(evidence)
explicitly: the product of the priors for root inputs, and the chain rule
P(e₁)·P(e₂ | e₁)·… for any other evidence. `infer()` and `query_batch()` also emit a
`UserWarning` naming the observed states with prior 0 and the number of affected
pixels or points, since a zero prior on an input is often a modelling slip.

## Output

[`InferenceResult`][geobn.InferenceResult] holds:

- `probabilities` — dict mapping each query node to a `(H, W, n_states)` float32 array
- `state_names` — ordered state labels per query node
- `crs`, `transform` — spatial metadata for the output grid

From an `InferenceResult` you can:

- `result.entropy("node")` — Shannon entropy map (bits), shape (H, W)
- summary maps: `expected_value`, `std`, `mode`, `mode_probability`, `exceedance`,
  `ignorance`, `quantile` (see [InferenceResult](api/result.md#summary-layers))
- `result.to_geotiff("out/", layers={...})` — write multi-band GeoTIFFs, plus optional extra layers
- `result.to_xarray(layers={...})` — return an xarray Dataset
- `result.show_map()` — interactive Leaflet map

## Real-time / repeated inference

When running inference repeatedly (e.g. updating weather forecasts while terrain stays
fixed), geobn provides two optimisation tiers:

**Tier 1 — `bn.freeze(*nodes)`**: marks nodes as static. The discrete index array
is computed once on the first `infer()` call and cached for all subsequent calls.
The pgmpy `VariableElimination` object is also cached.

**Tier 2 — `bn.precompute(query)`**: solves all ∏ n_states combinations once (via a
single joint query per query node) and stores a numpy lookup table. Subsequent
`infer()` calls use O(H×W) fancy indexing — zero pgmpy queries per call. Best for
real-time dashboards with a fixed BN structure.

Call `bn.clear_cache()` to reset all caches if inputs change.
