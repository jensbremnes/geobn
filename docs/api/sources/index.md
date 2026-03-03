# Data Sources — Overview

All data sources inherit from `DataSource` and implement a single method:

```python
def fetch(self, grid: GridSpec | None = None) -> RasterData:
    ...
```

`RasterData` is a named tuple `(array: np.ndarray, crs: str | None, transform: Affine | None)`.
No rasterio objects are ever exposed outside a source module.

## Source catalogue

| Source | Use case | Extra required |
|--------|----------|---------------|
| [`ArraySource`](core.md) | In-memory numpy array | — |
| [`ConstantSource`](core.md) | Single scalar broadcast over the grid | — |
| [`RasterSource`](core.md) | Local GeoTIFF file | `[io]` |
| [`URLSource`](core.md) | Remote GeoTIFF via HTTP | `[io]` |
| [`OpenMeteoSource`](weather.md) | Historical/forecast weather (Open-Meteo) | — |
| [`METOceanForecastSource`](weather.md) | Wave height, current speed (MET Norway) | — |
| [`METLocationForecastSource`](weather.md) | Wind speed/direction (MET Norway) | — |
| [`KartverketDTMSource`](terrain.md) | Norwegian 10 m Digital Terrain Model | `[io]` |
| [`EMODnetBathymetrySource`](terrain.md) | European seabed depth | `[io]` |
| [`EMODnetShippingDensitySource`](terrain.md) | Historical vessel traffic density | `[io]` |
| [`WCSSource`](wcs.md) | Generic OGC WCS endpoint | `[io]` |
| [`CopernicusMarineSource`](ocean.md) | CMEMS ocean model data | `[ocean]` |
| [`HubOceanSource`](ocean.md) | HubOcean STAC catalog | `[ocean]` |
| [`BarentswatchAISSource`](ocean.md) | Live/historical AIS vessel positions | — |

## Grid-aware vs self-contained sources

**Self-contained** sources (`ArraySource`, `ConstantSource`, `RasterSource`, `URLSource`)
ignore the `grid` argument. They carry their own spatial metadata.

**Grid-aware** sources need the reference grid to determine what geographic area to
query. They call `grid.extent_wgs84()` to obtain `(lon_min, lat_min, lon_max, lat_max)`
before making API or WCS requests.

Grid-aware sources are: `OpenMeteoSource`, `WCSSource`, `KartverketDTMSource`,
`EMODnetBathymetrySource`, `EMODnetShippingDensitySource`, `METOceanForecastSource`,
`METLocationForecastSource`, `CopernicusMarineSource`, `BarentswatchAISSource`,
`HubOceanSource`.

## DataSource ABC

::: geobn.sources.DataSource
    options:
      members:
        - fetch
