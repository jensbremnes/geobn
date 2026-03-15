# Real-time inference optimisation

geobn includes two tiers of optimisation for repeated or latency-sensitive inference,
such as continuously updating a dashboard as sensor values change.

---

## The problem

By default, every call to `bn.infer()` fetches, aligns, and discretises **all**
inputs, then runs one pgmpy `VariableElimination` query per unique evidence
combination.  For a 1000×1000 grid with terrain sources that never change, this
is wasteful: DEM fetch + reprojection + discretisation can take several seconds
even though the result is the same every time.

---

## Tier 1 — Freeze static nodes

Use [`freeze()`][geobn.GeoBayesianNetwork.freeze] to mark nodes whose input data
will not change between `infer()` calls:

```python
bn.freeze("slope_angle", "aspect")        # declare terrain as static

result = bn.infer(query=["avalanche_risk"])
# First call: fetches terrain normally, caches the discrete index array.

bn.set_input("recent_snow", geobn.ConstantSource(35.0))
result = bn.infer(query=["avalanche_risk"])
# Second call: terrain array reused from cache — no fetch, no reprojection.
```

What is cached after the first call:

- The discrete integer index array for every frozen node.
- The reference `GridSpec` (so it is not re-derived from sources).
- The pgmpy `VariableElimination` engine (constructed once, reused).

If the underlying data of a frozen node actually changes (e.g. you swap out the
DEM source), call [`clear_cache()`][geobn.GeoBayesianNetwork.clear_cache] to
discard the stale arrays before the next `infer()`.

---

## Tier 2 — Precompute all evidence combinations

When **all** inputs are effectively static and only a few dynamic scalar inputs
change, pre-run the entire state-space combinatorial product once and store the
results as a numpy lookup table:

```python
bn.precompute(query=["avalanche_risk"])
# One-time cost: ∏ n_states_i pgmpy queries.
# For a 3×2×3×3 state space this is 54 queries (< 1 second).

result = bn.infer(query=["avalanche_risk"])
# Zero pgmpy calls — O(H×W) numpy fancy indexing only.
```

After `precompute()`, each `infer()` call replaces every pgmpy query with a
single numpy index operation: the (H, W) discrete index grids are used directly
to look up the pre-filled probability table.

!!! note
    The table path activates automatically when `infer()` is called with the same
    `query` list that was passed to `precompute()`.  Calling `infer()` with a
    different query falls back to the normal VE path.

---

## Offline persistence — save and load the table

For robotics and edge-deployment workflows where pgmpy must not run at deploy
time, you can build the lookup table on a workstation and ship only the numpy
archive:

**Offline (workstation):**

```python
import geobn

bn = geobn.load("model.bif")
bn.set_input("slope", geobn.RasterSource("slope.tif"))
bn.set_input("rainfall", geobn.ConstantSource(50.0))
bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

bn.precompute(query=["fire_risk"])          # one-time cost: all state combos
bn.save_precomputed("fire_risk_table.npz")  # portable .npz — ship this file
```

**Runtime (robot / edge device):**

```python
import geobn

bn = geobn.load("model.bif")
bn.set_input("slope", geobn.RasterSource("slope.tif"))
bn.set_input("rainfall", geobn.ConstantSource(50.0))
bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

bn.load_precomputed("fire_risk_table.npz")  # no pgmpy inference — loads numpy archive
result = bn.infer(query=["fire_risk"])       # O(H×W) table lookup, zero pgmpy calls
```

!!! note
    `load_precomputed()` validates that the file's node order and array shapes
    match the current BN and discretization configuration.  A `ValueError` is
    raised if there is a mismatch; a `FileNotFoundError` is raised if the file
    does not exist.

---

## Combining both tiers

The tiers stack naturally:

```python
bn.freeze("slope_angle", "aspect")
bn.precompute(query=["avalanche_risk"])

# Subsequent infer() calls:
#   - Skip fetch/align/discretise for frozen terrain nodes (Tier 1)
#   - Skip all pgmpy queries (Tier 2)
result = bn.infer(query=["avalanche_risk"])
```

---

## Cache lifecycle

| Trigger | Effect |
|---------|--------|
| `bn.freeze(*nodes)` called with a **different** node set | `clear_cache()` is called automatically |
| `bn.set_grid(...)` | Cache cleared automatically |
| `bn.clear_cache()` | All caches discarded (frozen arrays, VE engine, inference table) |

---

## API reference

See the full method signatures in the [GeoBayesianNetwork](network.md) reference:

- [`freeze(*node_names)`][geobn.GeoBayesianNetwork.freeze]
- [`precompute(query)`][geobn.GeoBayesianNetwork.precompute]
- [`save_precomputed(path)`][geobn.GeoBayesianNetwork.save_precomputed]
- [`load_precomputed(path)`][geobn.GeoBayesianNetwork.load_precomputed]
- [`clear_cache()`][geobn.GeoBayesianNetwork.clear_cache]
