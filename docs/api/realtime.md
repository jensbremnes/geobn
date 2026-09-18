# Real-time inference optimisation

For repeated or latency-sensitive inference, such as a dashboard that updates as
sensor values change, geobn can cache static inputs and precompute results.

## The problem

By default every `bn.infer()` call fetches, aligns and discretises all inputs, then
runs batched pgmpy inference over the distinct evidence combinations. On a 1000×1000
grid with terrain that never changes, most of this is wasted: fetching, reprojecting
and discretising the DEM can take several seconds and gives the same result every
time.

## Freezing static inputs

[`freeze()`][geobn.GeoBayesianNetwork.freeze] marks nodes whose input data won't
change between `infer()` calls:

```python
bn.freeze("slope_angle", "aspect")        # terrain is static

result = bn.infer(query=["avalanche_risk"])
# First call: fetches terrain as usual and caches the discrete index array.

bn.set_input("recent_snow", geobn.ConstantSource(35.0))
result = bn.infer(query=["avalanche_risk"])
# Second call: terrain comes from the cache, with no fetch or reprojection.
```

After the first call geobn keeps:

- the discrete index array of every frozen node
- the reference `GridSpec`, so it isn't derived from the sources again
- the pgmpy `VariableElimination` engine

If the data behind a frozen node does change (say you swap the DEM source), call
[`clear_cache()`][geobn.GeoBayesianNetwork.clear_cache] before the next `infer()`.

## Precomputing a lookup table

If the network and discretization stay fixed and only input values change, you can
solve every combination of evidence states once and keep the results in a numpy
table:

```python
bn.precompute(query=["avalanche_risk"])
# One joint pgmpy query per query node covers every evidence combination,
# so this is fast even for large state spaces.

result = bn.infer(query=["avalanche_risk"])
# No pgmpy calls, only O(H×W) numpy fancy indexing.
```

`infer()` then uses the (H, W) discrete index grids to look up the probabilities
directly in the table.

!!! note
    The table is only used when `infer()` gets the same `query` list that was passed
    to `precompute()`. Any other query goes through normal variable elimination.

## Saving and loading the table

In robotics and edge deployments you may not want pgmpy to run on the target at all.
Build the table on a workstation and ship only the numpy archive.

On the workstation:

```python
import geobn

bn = geobn.load("model.bif")
bn.set_input("slope", geobn.RasterSource("slope.tif"))
bn.set_input("rainfall", geobn.ConstantSource(50.0))
bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

bn.precompute(query=["fire_risk"])
bn.save_precomputed("fire_risk_table.npz")  # ship this file
```

On the robot or edge device:

```python
import geobn

bn = geobn.load("model.bif")
bn.set_input("slope", geobn.RasterSource("slope.tif"))
bn.set_input("rainfall", geobn.ConstantSource(50.0))

# The discretizations are stored in the file, so set_discretization() isn't needed
bn.load_precomputed("fire_risk_table.npz")
result = bn.infer(query=["fire_risk"])       # table lookup, no pgmpy calls
```

!!! note
    The file records which model it was built from, and `load_precomputed()`
    checks it against the current BN:

    - The evidence nodes must be the current inputs. They may be registered in a
      different order; the table axes are reordered to match.
    - The BN state names and a hash of the model (structure, states and CPD values)
      must match, so a table from another model won't load.
    - Discretizations saved with the table (breakpoints, labels, `out_of_range`) are
      restored for inputs that have none set. If an input already has one, it must
      match the saved one.

    A mismatch raises `ValueError`, and a missing file raises `FileNotFoundError`.
    Files saved by older geobn versions still load with a `UserWarning`, but only
    their evidence nodes and array shapes can be checked. Run `precompute()` and
    `save_precomputed()` again to upgrade them.

## Point queries

Once a table exists (from `precompute()` or `load_precomputed()`), you can read
posteriors for single points without sources, grids or `infer()`. Evidence values can
be numbers, which are discretized with each node's breakpoints, or state names:

```python
bn.query_point({"slope": 35.0, "rainfall": "high"})
# {'fire_risk': {'low': 0.2, 'medium': 0.3, 'high': 0.5}}
```

For many points, pass one sequence per node (lists, tuples or arrays of equal length
K). A single value applies to every point:

```python
probs = bn.query_batch({
    "slope": [5.0, 20.0, 45.0],
    "rainfall": "high",
})["fire_risk"]
# (3, 3) float32 array, one row per point, states in BN order
```

A NaN value gives NaN probabilities for that point, as a NoData pixel does in
`infer()`. Every input node registered with `set_input()` needs a value, because the
table can't leave one out.

## Using both together

Freezing and precomputing can be combined:

```python
bn.freeze("slope_angle", "aspect")
bn.precompute(query=["avalanche_risk"])

# Later infer() calls skip fetching, aligning and discretising the frozen
# terrain nodes, and make no pgmpy queries.
result = bn.infer(query=["avalanche_risk"])
```

## Cache lifecycle

| Trigger | Effect |
|---------|--------|
| `bn.freeze(*nodes)` called with a **different** node set | `clear_cache()` is called automatically |
| `bn.set_grid(...)` | Cache cleared automatically |
| `bn.clear_cache()` | All caches discarded (frozen arrays, VE engine, inference table) |

## API reference

Full signatures are in the [GeoBayesianNetwork](network.md) reference:

- [`freeze(*node_names)`][geobn.GeoBayesianNetwork.freeze]
- [`precompute(query)`][geobn.GeoBayesianNetwork.precompute]
- [`save_precomputed(path)`][geobn.GeoBayesianNetwork.save_precomputed]
- [`load_precomputed(path)`][geobn.GeoBayesianNetwork.load_precomputed]
- [`query_point(evidence)`][geobn.GeoBayesianNetwork.query_point]
- [`query_batch(evidence)`][geobn.GeoBayesianNetwork.query_batch]
- [`clear_cache()`][geobn.GeoBayesianNetwork.clear_cache]
