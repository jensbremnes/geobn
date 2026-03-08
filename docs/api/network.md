# GeoBayesianNetwork

The primary user-facing class. Load a `.bif` file with `geobn.load()`, attach data
sources to evidence nodes, configure discretization, and call `infer()`.

## Factory function

::: geobn.load
    options:
      show_root_heading: true

## GeoBayesianNetwork class

::: geobn.GeoBayesianNetwork
    options:
      members:
        - set_input
        - fetch_raw
        - set_discretization
        - set_grid
        - freeze
        - precompute
        - clear_cache
        - infer

---

## Real-time optimisation

For repeated inference with static terrain inputs or pre-computed state tables,
see the dedicated [Real-time optimisation](realtime.md) guide covering
`freeze()`, `precompute()`, and `clear_cache()`.
