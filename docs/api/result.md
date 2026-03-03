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
        - to_geotiff
        - to_xarray
        - show_map

## Band layout in GeoTIFF output

When calling `to_geotiff(output_dir)`, one multi-band GeoTIFF is written per query
node:

| Band | Content |
|------|---------|
| 1 … N | P(state_i \| evidence) for each state i |
| N + 1 | Shannon entropy (bits) |

Band descriptions in the file metadata contain the state label or `"entropy"`.

## Example

```python
result = bn.infer(query=["avalanche_risk"])

# Inspect probabilities
probs = result.probabilities["avalanche_risk"]  # (H, W, 3)
print(probs.shape)  # (80, 240, 3) for an 80×240 grid with 3 states

# Entropy map
ent = result.entropy("avalanche_risk")  # (H, W)

# Export
result.to_geotiff("output/")   # writes avalanche_risk.tif
ds = result.to_xarray()        # xarray Dataset with (state, y, x) dims
result.show_map()              # opens browser
```
