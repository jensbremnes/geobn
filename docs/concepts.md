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

[`geobn.load()`][geobn.load] picks a reader from the file extension:

| Extension | Format | Typically written by |
|---|---|---|
| `.bif` | BIF (Bayesian Interchange Format) | bnlearn, most exporters |
| `.xmlbif` | XMLBIF | Weka, JavaBayes |
| `.net` | Hugin NET | Hugin, GeNIe |
| `.xdsl` | GeNIe XDSL | GeNIe / SMILE |
| `.uai` | UAI | UAI competition files |
| `.xml` | sniffed from the root element | see below |

```python
bn = geobn.load("my_model.bif")
bn = geobn.load("my_model.xdsl")   # same call, GeNIe file
```

All readers ship with pgmpy, so no extra dependency is needed. Files are decoded as
UTF-8, falling back to the platform default encoding.

A bare `.xml` file is resolved from its root element: `<BIF>` is read as XMLBIF,
`<smile>` as GeNIe XDSL and `<ANALYSISNOTEBOOK>` as Microsoft XBN. Any other root
element raises `ValueError`.

`load()` also runs pgmpy's `check_model()`, so a truncated file or a CPT whose columns
do not sum to 1 fails at load time rather than deep inside `infer()`.

!!! warning "UAI files carry no names"
    The UAI format stores no variable or state names, so nodes come back as `var_0`,
    `var_1`, … and states as integers, in file order. `geobn.load()` emits a
    `UserWarning` and `set_input()` / `set_discretization()` must use those names.

!!! note "Netica `.dne`"
    Not supported — pgmpy has no Netica reader. Export to `.net` or `.bif` from Netica.

### Using a pgmpy model directly

`GeoBayesianNetwork` accepts any pgmpy `DiscreteBayesianNetwork`, so a model built in
code, fitted with a pgmpy estimator, or read with a reader `load()` does not dispatch on
can be used without going through a file:

```python
from pgmpy.readwrite import XBNReader

bn = geobn.GeoBayesianNetwork(XBNReader("model.dat").get_model())
```

Unlike `load()`, this does not run `check_model()`.

## Attaching data sources

Any node in the BN can be an evidence variable. Usually these are the root nodes (no
parents), but an intermediate or leaf node can be observed just as well. Attach a
[`DataSource`][geobn.sources.DataSource] to each one:

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

### Evidence on intermediate nodes

Evidence is not limited to the nodes at the top of the network. A source attached to an
intermediate or leaf node observes that node, and the evidence flows in both directions:
down to its descendants and up to its parents. A dataset that measures an intermediate
concept directly can therefore be used in place of the layers behind it.

In the bundled Lyngen model, `terrain_factor` is an intermediate node with parents
`slope_angle`, `sun_exposure` and `forest_cover`. Observing it and querying a parent
gives a distribution over slope classes:

```python
bn = geobn.load("avalanche_risk.bif")
bn.set_input("terrain_factor", geobn.RasterSource("terrain_class.tif"))
bn.set_discretization("terrain_factor", [0, 1, 2, 3], ["low", "medium", "high"])
result = bn.infer(query=["slope_angle"])        # P(slope | observed terrain)
```

`slope_angle` has a uniform prior in that model, so the observation shifts it
substantially:

| observed | flat | gentle | steep | extreme |
|---|---|---|---|---|
| *(prior)* | 0.250 | 0.250 | 0.250 | 0.250 |
| `terrain_factor="low"` | 0.422 | 0.375 | 0.123 | 0.081 |
| `terrain_factor="high"` | 0.006 | 0.039 | 0.372 | 0.583 |

This is ordinary Bayesian inference read in the less common direction, and its accuracy
depends entirely on the CPD being inverted. It is useful for filling gaps and for
cross-checking one layer against another, and it yields a distribution over states
rather than a measured value.

Root inputs are marginally independent, so any combination of them is possible. Inputs
on nodes with parents are not: observing a node together with its parent in a
combination the CPD gives probability zero leaves the posterior undefined. Those pixels
become NaN with a `UserWarning`, as with any other impossible evidence (see below).

A node cannot be both an input and a query node. `infer()`, `precompute()` and
`load_precomputed()` raise `ValueError` naming it, before any data is fetched.

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

### Breakpoints from the data

`geobn.breakpoints` computes a breakpoint list from an array, for the two standard
classification schemes:

```python
arr = bn.fetch_raw(geobn.RasterSource("slope.tif"))

geobn.breakpoints.quantile(arr, 3)        # bins with equal pixel counts
geobn.breakpoints.equal_interval(arr, 3)  # bins of equal width
```

`bn.suggest_breakpoints(node)` does the same for a node that already has a source,
taking the bin count from the node's states in the BN:

```python
bn.set_input("slope_angle", geobn.RasterSource("slope.tif"))
bn.set_discretization("slope_angle", bn.suggest_breakpoints("slope_angle"))
```

The outer breakpoints are the data's minimum and maximum, which makes them the valid
range as well. Pass `bounds=` to pin a node's physical range instead, so later data
beyond the values at hand is still in range:

```python
bn.suggest_breakpoints("slope_angle", bounds=(0, 90))
```

Breakpoints computed this way describe the data they were computed from. They fit the
case where a model's conditional probabilities are authored alongside its breakpoints;
a CPT elicited against fixed thresholds — "steep means over 30°" — needs those
thresholds.

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

pgmpy is imported only when a network is built or loaded, and when inference runs
through it. `import geobn` and the parts of the library that do not touch a network
(sources, grids, discretization, breakpoint helpers, `InferenceResult`) do not import
it. A `GeoBayesianNetwork` wraps a pgmpy model, so using a saved table still requires
pgmpy to be installed.

Call `bn.clear_cache()` to reset all caches if inputs change.
