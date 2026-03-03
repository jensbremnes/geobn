# geobn — Claude Code Instructions

## Git workflow (mandatory)

After completing any meaningful unit of work — a new feature, a set of tests, a bug fix, a refactor — **commit and push to GitHub immediately**. Never accumulate large batches of unrelated changes in a single commit.

```bash
git add <specific files>
git commit -m "concise present-tense description"
git push origin main
```

Rules:
- Stage only the files relevant to the change (never `git add -A` blindly).
- Write concise, present-tense commit messages: `"add WCSSource"`, `"fix nodata sentinel in KartverketDTMSource"`.
- Push after every commit. The remote on GitHub (`https://github.com/jensebr/geobn.git`) is the canonical backup.
- Always run `uv run pytest tests/ -v` before committing. Do not commit if tests are red.

---

## Project overview

`geobn` is a Python library for **pixel-wise Bayesian network inference over geospatial data**. The user loads a Bayesian network (`.bif` file), attaches geographic data sources to evidence nodes, defines discretization rules, and runs inference to produce posterior probability rasters and entropy maps.

```
DataSources → align to reference grid → discretize → BN inference → InferenceResult
```

- Python ≥ 3.13, managed with `uv`.
- Core deps: `pgmpy`, `numpy`, `requests`, `pyproj`, `affine`.
- Optional `[io]` extra: `rasterio` — required for GeoTIFF read/write.
- Optional `[viz]` extra: `folium>=0.14`, `matplotlib>=3.0` — required for `result.show_map()`.
- Optional `[ocean]` extra: `copernicusmarine>=2.0`, `pystac-client>=0.7`.
- Optional `[full]` extra: `[io,viz]` + `xarray>=2024.1` — recommended for office/analysis use.
- Dev extra installs `geobn[full]` + `pytest`.

---

## Repository layout

```
src/geobn/
  __init__.py          Public API re-exports (all sources + network + result)
  _types.py            RasterData NamedTuple(array, crs, transform)
  _io.py               rasterio write helpers — lazy import, isolated
  _viz.py              show_map() + helpers — folium [viz] optional dep
  network.py           GeoBayesianNetwork class + load() factory
  grid.py              GridSpec, align_to_grid(), _reproject(), _bilinear_resample()
  discretize.py        DiscretizationSpec, discretize_array()
  inference.py         run_inference() — unique-combo batching, shannon_entropy()
  result.py            InferenceResult → to_geotiff() / to_xarray() / show_map()
  sources/
    _base.py             DataSource ABC: fetch(grid=None) -> RasterData
    array_source.py      ArraySource — in-memory numpy array
    constant_source.py   ConstantSource — scalar broadcast; crs=None, transform=None
    raster_source.py     RasterSource — local file via rasterio
    url_source.py        URLSource — remote GeoTIFF via rasterio.MemoryFile
    openmeteo_source.py  OpenMeteoSource — Open-Meteo archive/forecast API
    wcs_source.py        WCSSource — generic OGC WCS (v2.0.1 + v1.1.1 fallback)
    kartverket_source.py KartverketDTMSource — Norwegian DTM via Kartverket WCS
    emodnet_source.py    EMODnetBathymetrySource — European seabed via EMODnet WCS
    met_norway_source.py METOceanForecastSource + METLocationForecastSource
    copernicus_source.py CopernicusMarineSource — CMEMS via copernicusmarine SDK
    barentswatch_source.py BarentswatchAISSource — AIS vessel tracking, OAuth2
    hubocean_source.py   HubOceanSource — STAC catalog + xarray
tests/
  conftest.py            Shared fixtures (fire_risk_model, reference_transform, arrays)
  test_discretize.py
  test_grid.py
  test_inference.py
  test_integration.py
  test_result.py
  test_sources.py
  test_new_sources.py    Tests for all 8 new sources (mocked, no network)
examples/
  lyngen_alps/           Avalanche risk: Kartverket DTM + ConstantSource (flagship intro example)
  maritime_norway/       Maritime nav risk: EMODnet + MET Norway + AIS density
tools/
  xdsl_to_bif.py         Converter: GeNIe .xdsl → standard .bif
private/                 Gitignored; local experiments only
```

---

## Core architecture decisions

### RasterData
All sources return `RasterData(array: np.ndarray, crs: str|None, transform: Affine|None)`. No rasterio objects ever leave a source module. `ConstantSource` sets `crs=None, transform=None` as a sentinel — `align_to_grid()` broadcasts it.

### Grid alignment
The first georeferenced source registered via `set_input()` sets the reference `GridSpec`. Override with `bn.set_grid(crs, resolution, extent)`. All other sources are reprojected to this grid using pure numpy + pyproj (no rasterio) bilinear interpolation.

### NoData
`NaN` throughout. Any pixel that is NaN in any input is excluded from inference and stays NaN in all output bands.

### Inference batching
`np.unique(..., axis=0, return_inverse=True)` groups pixels by unique evidence combinations. One `pgmpy.inference.VariableElimination` query runs per unique combo, not per pixel.

### Lazy imports
Optional dependencies (`rasterio`, `copernicusmarine`, `pystac_client`) are imported inside `fetch()`, never at module level. Missing deps raise `ImportError` with an install hint (`pip install geobn[io]` or `pip install geobn[ocean]`).

### Credential/grid checks before lazy imports
In sources with both optional deps and required credentials (CopernicusMarineSource, HubOceanSource), validate credentials and grid **before** the lazy import so that missing-credential errors are raised even when the optional package is absent.

### pgmpy version
pgmpy ≥ 1.0 uses `DiscreteBayesianNetwork`; `BIFReader.get_model()` returns it automatically. Never use the deprecated `BayesianNetwork`.

---

## Data source patterns

### Grid-aware sources (require grid in fetch)
`OpenMeteoSource`, `WCSSource`, `KartverketDTMSource`, `EMODnetBathymetrySource`, `METOceanForecastSource`, `METLocationForecastSource`, `CopernicusMarineSource`, `BarentswatchAISSource`, `HubOceanSource`.

All call `grid.extent_wgs84()` to get `(lon_min, lat_min, lon_max, lat_max)`.

### Point-sampling sources
`OpenMeteoSource`, `METOceanForecastSource`, `METLocationForecastSource` sample an N×N lat/lon grid over the bbox and return a coarse EPSG:4326 raster; `align_to_grid()` bilinearly resamples to the reference grid. Sleep 0.05 s between calls.

### WCS-based sources
`WCSSource` builds a GetCoverage request (v2.0.1 uses `SUBSET=Lat(…)&SUBSET=Long(…)`; v1.1.1 uses `BBOX=`), receives GeoTIFF bytes, parses with `rasterio.MemoryFile`. `KartverketDTMSource` and `EMODnetBathymetrySource` compose `WCSSource` and apply nodata sentinel replacement.

### Nodata sentinels
- Kartverket DTM: `array < −500` or `array > 9000` → NaN.
- EMODnet Bathymetry: `array > 9000` or `array < −15000` → NaN. Negative values (depth) are valid.

### MET Norway User-Agent
MET Norway ToS requires `User-Agent: geobn/0.1` on all requests. Both MET sources set this header.

### BarentswatchAISSource
OAuth2 client-credentials flow → Bearer token → AIS REST API for bbox. Vessel lat/lon → grid CRS via `pyproj.Transformer`; inverse grid affine → pixel indices. Metrics: `"density"` (vessels/km²), `"count"`, `"speed"` (mean knots).

---

## Development commands

```bash
# Install in editable mode with all dev dependencies
uv pip install -e ".[dev]"

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_new_sources.py -v

# Run examples
uv run python examples/lyngen_alps/run_example.py
uv run python examples/maritime_norway/run_example.py
```

---

## Testing conventions

- All tests in `tests/`; fixtures in `conftest.py`.
- Use `unittest.mock.patch("requests.get", ...)` to mock HTTP sources.
- Use `patch.dict(sys.modules, {"copernicusmarine": None})` to simulate missing optional deps.
- Use `pytest.importorskip("rasterio")` to skip rasterio-dependent tests when it is absent.
- No real network calls in tests. Tests must pass fully offline.
- New sources need tests covering: missing grid, invalid constructor args, HTTP errors, nodata handling, successful fetch with mocked response.
- Run `uv run pytest tests/ -v` before every commit. All tests must be green.

---

## Adding a new source

1. Create `src/geobn/sources/my_source.py` following the DataSource ABC.
2. Export from `src/geobn/sources/__init__.py` and `src/geobn/__init__.py`.
3. Add tests to `tests/test_new_sources.py` (or a new test file).
4. If the source needs an optional package, add it to the appropriate extra in `pyproject.toml`.
5. Update `CLAUDE.md` (this file) and `memory/MEMORY.md`.
6. Commit and push.
