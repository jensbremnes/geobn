# How it works

## Pipeline overview

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌──────────────────┐
│ DataSource   │────►│ align_to_grid()  │────►│ discretize() │────►│ BN inference     │
│ (per node)   │     │ reproject + resample   │ continuous → │     │ VariableElim.    │
└──────────────┘     │ to reference grid│     │ BN states    │     │ per unique combo │
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
bn.set_input("slope_angle", geobn.KartverketDTMSource())
bn.set_input("recent_snow", geobn.ConstantSource(30.0))
```

The first call to `set_input` with a georeferenced source establishes the **reference
grid** (CRS, resolution, extent). All other sources are reprojected to this grid
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

`breakpoints` must have `len(labels) + 1` values. Pixels outside `[breakpoints[0],
breakpoints[-1]]` become NaN (no inference).

## NaN / NoData propagation

NaN values propagate strictly: if **any** input pixel is NaN, that pixel is excluded
from inference and all output bands for that pixel are NaN.

This means:
- Pixels outside WCS coverage → NaN inputs → NaN outputs
- Sea pixels in a land DEM → NaN depth → NaN output
- Invalid sensor readings → NaN evidence → NaN posteriors

## Inference batching

Running one pgmpy `VariableElimination.query()` per pixel is prohibitively slow for
large rasters. geobn uses `np.unique(..., axis=0, return_inverse=True)` to group all
pixels by their unique discrete evidence combination. One pgmpy query runs per unique
combination, and the result is scattered back to the original pixel positions.

For a 500×500 grid with 3 evidence nodes (3 states each), there are only 27 possible
unique combinations regardless of grid size.

## Output

[`InferenceResult`][geobn.InferenceResult] holds:

- `probabilities` — dict mapping each query node to a `(H, W, n_states)` float32 array
- `state_names` — ordered state labels per query node
- `crs`, `transform` — spatial metadata for the output grid

From an `InferenceResult` you can:

- `result.entropy("node")` — Shannon entropy map (bits), shape (H, W)
- `result.to_geotiff("out/")` — write multi-band GeoTIFFs (requires `[io]`)
- `result.to_xarray()` — return an xarray Dataset (requires `[full]`)
- `result.show_map()` — interactive Leaflet map (requires `[viz]`)

## Lazy imports

Optional dependencies (`rasterio`, `copernicusmarine`, `pystac_client`, `folium`,
`xarray`) are imported inside `fetch()` or the method that needs them — never at module
level. If the dependency is missing, a clear `ImportError` with an install hint is
raised:

```
ImportError: rasterio is required for RasterSource.
Install it with: pip install "geobn[io]"
```

## Real-time / repeated inference

When running inference repeatedly (e.g. updating weather forecasts while terrain stays
fixed), geobn provides two optimisation tiers:

**Tier 1 — `bn.freeze(*nodes)`**: marks nodes as static. The discrete index array
is computed once on the first `infer()` call and cached for all subsequent calls.
The pgmpy `VariableElimination` object is also cached.

**Tier 2 — `bn.precompute(query)`**: runs all ∏ n_states combinations once and stores
a numpy lookup table. Subsequent `infer()` calls use O(H×W) fancy indexing — zero pgmpy
queries per call. Best for real-time dashboards with a fixed BN structure.

Call `bn.clear_cache()` to reset all caches if inputs change.
