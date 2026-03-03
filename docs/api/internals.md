# Internals

!!! note "For contributors, not end users"
    This page documents the internal modules used by `GeoBayesianNetwork` and the
    data sources. Normal users interact exclusively with the public API described in
    [GeoBayesianNetwork](network.md), [InferenceResult](result.md), and the
    [Sources](sources/index.md) pages.

## Grid module (`grid.py`)

::: geobn.grid.GridSpec
    options:
      show_root_heading: true

::: geobn.grid.align_to_grid
    options:
      show_root_heading: true

## Discretization module (`discretize.py`)

::: geobn.discretize.DiscretizationSpec
    options:
      show_root_heading: true

::: geobn.discretize.discretize_array
    options:
      show_root_heading: true

## Inference module (`inference.py`)

::: geobn.inference.run_inference
    options:
      show_root_heading: true

::: geobn.inference.run_inference_from_table
    options:
      show_root_heading: true

::: geobn.inference.shannon_entropy
    options:
      show_root_heading: true

## Types (`_types.py`)

::: geobn._types.RasterData
    options:
      show_root_heading: true

## I/O helpers (`_io.py`)

::: geobn._io.write_geotiff
    options:
      show_root_heading: true

## Disk cache utilities (`sources/_cache.py`)

::: geobn.sources._cache._make_cache_path
    options:
      show_root_heading: true

::: geobn.sources._cache._load_cached
    options:
      show_root_heading: true

::: geobn.sources._cache._save_cached
    options:
      show_root_heading: true
