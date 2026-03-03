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
        - set_discretization
        - set_grid
        - freeze
        - precompute
        - clear_cache
        - infer
