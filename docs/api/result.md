# InferenceResult

The object returned by [`GeoBayesianNetwork.infer()`][geobn.GeoBayesianNetwork.infer].

Holds per-pixel probability distributions for one or more query nodes. The spatial
metadata (`crs`, `transform`) mirrors the reference grid used during inference.

## Class reference

::: geobn.InferenceResult
    options:
      members:
        - probabilities
        - state_names
        - crs
        - transform
        - entropy
        - expected_value
        - std
        - mode
        - mode_probability
        - exceedance
        - ignorance
        - quantile
        - to_geotiff
        - to_xarray
        - show_map

## Summary layers

Each method reduces the per-pixel distribution of one query node to a single
`(H, W)` float32 map. Pixels with NoData input are NaN. State indices follow the
order in `state_names[node]`, so order the states from least to most severe for
`exceedance` and `quantile` to make sense.

For a pixel with P(low, medium, high) = (0.2, 0.5, 0.3):

| Method | Meaning | Example |
|--------|---------|---------|
| `expected_value(node, values)` | Σ P(state) · value(state), e.g. expected loss | `[10, 50, 90]` → 54 |
| `std(node, values)` | Standard deviation of `values` | ≈ 28 |
| `mode(node)` | Index of the most probable state | 1 (medium) |
| `mode_probability(node)` | Probability of that state | 0.5 |
| `exceedance(node, state)` | P(node ≥ state) | `"medium"` → 0.8 |
| `ignorance(node, threshold)` | Mode where P(mode) ≥ threshold, else NaN | `0.6` → NaN |
| `quantile(node, q)` | Lowest state k with P(node ≤ k) ≥ q: the worst state at confidence q | `0.95` → 2 (high) |

`values` is either a list in state order or a dict keyed by state name
(`{"low": 10, "medium": 50, "high": 90}`), which guards against order mistakes.
geobn does not choose these numbers; they are your consequence scores or costs.

Pass summary maps (or any other `(H, W)` array on the result grid) to the exporters:

```python
score = result.expected_value("usv_risk", {"low": 10, "medium": 50, "high": 90})
result.to_geotiff("out/", layers={"risk_score": score})  # also writes out/risk_score.tif
ds = result.to_xarray(layers={"risk_score": score})      # adds a (y, x) variable
```

## Band layout in GeoTIFF output

When calling `to_geotiff(output_dir)`, one multi-band GeoTIFF is written per query
node:

| Band | Content |
|------|---------|
| 1 … N | P(state_i \| evidence) for each state i |
| N + 1 | Shannon entropy (bits) |

Band descriptions in the file metadata contain the state label or `"entropy"`.
Each extra layer passed via `layers=` is written as a single-band `{name}.tif`
whose band description is the layer name.

## Example

```python
result = bn.infer(query=["avalanche_risk"])

# Inspect probabilities
probs = result.probabilities["avalanche_risk"]  # (H, W, 3)
print(probs.shape)  # (80, 240, 3) for an 80×240 grid with 3 states

# Entropy map
ent = result.entropy("avalanche_risk")  # (H, W)

# P(avalanche_risk >= high)
p_high = result.exceedance("avalanche_risk", "high")

# Export
result.to_geotiff("output/")   # writes avalanche_risk.tif
ds = result.to_xarray()        # xarray Dataset with (state, y, x) dims
result.show_map()              # opens browser
```
