# Breakpoints

Helpers that compute the breakpoint list
[`set_discretization()`][geobn.GeoBayesianNetwork.set_discretization] takes, from the
data itself:

```python
arr = bn.fetch_raw(geobn.RasterSource("slope.tif"))
bn.set_discretization("slope_angle", geobn.breakpoints.quantile(arr, 4))
```

For a node that already has a source registered,
[`bn.suggest_breakpoints()`][geobn.GeoBayesianNetwork.suggest_breakpoints] does the
fetch and takes the bin count from the BN:

```python
bn.set_input("slope_angle", geobn.RasterSource("slope.tif"))
bn.set_discretization("slope_angle", bn.suggest_breakpoints("slope_angle"))
```

Both schemes return `n + 1` ascending edges. The interior values are the bin edges; the
outer pair defines the valid range `[first, last]`, so pass `bounds=` to pin a node's
physical range when later data may go beyond the values at hand.

## Choosing a scheme

| | `quantile` | `equal_interval` |
|---|---|---|
| Edges from | the distribution | the range |
| Bin occupancy | roughly equal | whatever the data gives |
| Bin width | varies | constant |
| Suits | skewed data, where equal-width bins would leave states empty | data spread evenly across its range, or thresholds that should read as round numbers |

Breakpoints computed from data describe the data they were computed from. They fit the
case where a model's conditional probabilities are authored alongside its breakpoints; a
CPT elicited against fixed thresholds needs those thresholds instead.

## Function reference

::: geobn.breakpoints.quantile
    options:
      show_root_heading: true

::: geobn.breakpoints.equal_interval
    options:
      show_root_heading: true
