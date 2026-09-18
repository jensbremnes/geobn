# How it works

## Pipeline overview

```
DataSource (per node)  →  align_to_grid()  →  discretize()  →  BN inference  →  InferenceResult
                          reproject and       continuous       batched VE        (H, W, n_states)
                          resample            → BN states      queries           + entropy map
```

## Loading a Bayesian network

geobn reads `.bif` files (Bayesian Interchange Format), which GeNIe, Netica and
bnlearn can all export. Load one with [`geobn.load()`][geobn.load]:

```python
bn = geobn.load("my_model.bif")
```

## Attaching data sources

Each root node (a node without parents) is an evidence variable. Attach a
[`DataSource`][geobn.sources.DataSource] to each of them:

```python
bn.set_input("slope_angle", geobn.WCSSource(url="https://example.com/wcs", layer="dtm"))
bn.set_input("recent_snow", geobn.ConstantSource(30.0))
```

At inference time geobn looks at the georeferenced sources and uses the one with the
finest resolution as the reference grid (CRS, resolution and extent). Pixel sizes are
compared in metres on the ground, measured at the centre of each grid. A 10 m UTM
raster therefore wins over a 0.001° WGS84 raster, even though 0.001 < 10 in CRS
units. Every other source is reprojected onto this grid. To choose the grid yourself,
call `bn.set_grid(crs, resolution, extent)` before inference.

## GridSpec and alignment

The reference grid is a `GridSpec(crs, transform, shape)`. Alignment uses bilinear
interpolation written with numpy and pyproj, without rasterio.

`ConstantSource` returns a 1×1 array with `crs=None`. `align_to_grid()` recognises it
and broadcasts it to the full grid shape.

## Discretization

The network works with discrete states, so every continuous input has to be mapped to
state names with `set_discretization()`:

```python
bn.set_discretization(
    "slope_angle",
    breakpoints=[0, 25, 40, 90],   # bin edges
    labels=["gentle", "steep", "extreme"],  # must match BN state names exactly
)
```

`breakpoints` needs `len(labels) + 1` values. A value that falls exactly on an interior
edge goes to the upper bin.

The first and last breakpoints give the valid range `[breakpoints[0], breakpoints[-1]]`
(inclusive). With the default `out_of_range="clip"`, values outside the range go to
the first or last state, so a 95° slope counts as `"extreme"`. With
`out_of_range="nan"` they are treated as NoData:

```python
bn.set_discretization(
    "slope_angle",
    breakpoints=[0, 25, 40, 90],
    labels=["gentle", "steep", "extreme"],
    out_of_range="nan",   # values < 0 or > 90 → NaN output
)
```

Use this when out-of-range values mean bad data (sensor errors, fill values) and not
extreme conditions.

## NaN / NoData propagation

If any input is NaN at a pixel, that pixel is skipped and every output band is NaN
there. In practice:

- pixels outside WCS coverage give NaN outputs
- sea pixels in a land DEM have no depth, and give NaN outputs
- invalid sensor readings give NaN posteriors
- values outside the breakpoint range give NaN posteriors when `out_of_range="nan"`
- `RasterSource`, `URLSource` and `WCSSource` convert pixels equal to the GeoTIFF's
  declared nodata value (or masked by its internal mask) to NaN

An `ArraySource` without `crs`/`transform` must either have exactly the grid shape or
hold a single value. Any other shape raises `ValueError` instead of being broadcast.

## Inference batching

One pgmpy `VariableElimination.query()` per pixel would be far too slow for a large
raster, and geobn never does that. It first counts how many distinct evidence
combinations the map contains, and then picks the cheapest strategy for that count.

### Step 1: group pixels by observed evidence combination

After discretization every pixel has one state index per evidence node, for example
`(slope=steep, rainfall=high)`. geobn groups the valid pixels by these combinations
and counts how many distinct ones occur on the map.

That count is usually much smaller than the number of possible combinations:

- It can't be larger than the number of valid pixels.
- Neighbouring pixels tend to share slope class, forest class and so on.
- A scalar input (`ConstantSource`, a live weather value) has only one state, so it
  doesn't add combinations.

A network with millions of possible combinations may produce only a few dozen on a
real map. The strategy depends only on the observed count, so a large theoretical
state space costs nothing by itself.

### Step 2: choose a strategy

With few observed combinations (200 or fewer), geobn runs one pgmpy
`VariableElimination` query per combination. A 500×500 grid with 3 evidence nodes of
3 states each has at most 27 combinations, whatever the grid size, and a few targeted
queries are the cheapest option.

How the query nodes are requested within one combination depends on the size of their
joint table (the product of their state counts). pgmpy computes the full joint over
all requested nodes before splitting it into marginals:

- If the joint has at most 20,000 cells, all query nodes go into one pgmpy call. This
  shares the elimination work and can be up to about 3× faster than separate calls.
- If it is larger, each query node gets its own call. The joint grows multiplicatively
  with each query node (19 nodes with 2–5 states each is 192 million cells), while
  separate calls grow linearly. In that case one call per node took 32 ms per
  combination, compared with 10 s for the combined call.

With many observed combinations, running thousands of near-identical queries repeats
the same propagation work. geobn instead asks pgmpy once for the joint distribution
P(query, e₁, …, eₖ) with no evidence. Normalising it along the query axis gives the
conditional table P(query | e₁, …, eₖ), which answers every combination at once:

```
P(query | e₁, …, eₖ) = P(query, e₁, …, eₖ) / P(e₁, …, eₖ)
```

Pixel results are then read from the table with numpy fancy indexing. With 10
evidence nodes of 3 states each (59,049 combinations) this takes well under a second,
where the per-combination loop would take minutes.

The joint table has one cell per combination of evidence states times the number of
query states. If that would take more than about 80 MB, geobn skips the joint query
and falls back to the per-combination loop. The loop is slower, but it only pays for
combinations that actually occur.

### Summary of regimes

| Observed combos | Table fits in memory? | Strategy | Cost |
|---|---|---|---|
| ≤ 200 | (irrelevant) | per-combination queries | milliseconds |
| many | yes | single joint query + table lookup | ~constant, sub-second |
| many | no | per-combination queries | proportional to observed combos |

Either way, the results are scattered back to the original pixel positions.

### Impossible evidence

If an evidence combination has probability zero (for example an input observed in a
state with prior 0), the posterior is undefined and every query node gets NaN, whichever
strategy is used. pgmpy alone would still return numbers for query nodes it can prune
away from the contradiction, so geobn checks P(evidence) itself. For root inputs this is
the product of the priors; for other evidence it uses the chain rule
P(e₁)·P(e₂ | e₁)·…. `infer()` and `query_batch()` also raise a `UserWarning` that
lists the observed states with prior 0 and how many pixels or points they affect, since
a zero prior on an input is often a modelling mistake.

## Output

[`InferenceResult`][geobn.InferenceResult] holds:

- `probabilities`: dict mapping each query node to a `(H, W, n_states)` float32 array
- `state_names`: ordered state labels per query node
- `crs`, `transform`: spatial metadata for the output grid

From an `InferenceResult` you can get:

- `result.entropy("node")`: Shannon entropy map in bits, shape (H, W)
- summary maps: `expected_value`, `std`, `mode`, `mode_probability`, `exceedance`,
  `ignorance`, `quantile` (see [InferenceResult](api/result.md#summary-layers))
- `result.to_geotiff("out/", layers={...})`: multi-band GeoTIFFs, plus optional extra layers
- `result.to_xarray(layers={...})`: an xarray Dataset
- `result.show_map()`: an interactive Leaflet map

## Real-time / repeated inference

When you run inference over and over, for example with new weather forecasts on fixed
terrain, there are two ways to save work.

`bn.freeze(*nodes)` marks nodes as static. Their discrete index arrays are computed on
the first `infer()` call and reused afterwards. The pgmpy `VariableElimination` object
is cached as well.

`bn.precompute(query)` solves all ∏ n_states combinations once, using a single joint
query per query node, and stores the result as a numpy lookup table. Later `infer()`
calls only do O(H×W) fancy indexing and make no pgmpy queries. This suits live
dashboards where the network structure doesn't change.

Call `bn.clear_cache()` to reset all caches if inputs change. See
[Real-time optimisation](api/realtime.md) for more.
