# Weather Sources

These sources query meteorological APIs and return coarse rasters that are bilinearly
resampled to the reference grid by `align_to_grid()`.

All weather sources are **grid-aware** — they require a reference grid at `fetch()` time
(provided automatically by `GeoBayesianNetwork.infer()`).

## OpenMeteoSource

::: geobn.OpenMeteoSource
    options:
      show_root_heading: true

No API key required. Uses the free [Open-Meteo](https://open-meteo.com/) archive API.
A 0.05 s sleep is inserted between point requests to respect rate limits.

**Example:**

```python
# Fetch yesterday's mean temperature across the study area
source = geobn.OpenMeteoSource(
    variable="temperature_2m_mean",
    date="2024-01-15",
    sample_points=5,
)
bn.set_input("air_temp", source)
```

---

## METOceanForecastSource

::: geobn.METOceanForecastSource
    options:
      show_root_heading: true

Uses the [MET Norway OceanForecast 2.0 API](https://api.met.no/weatherapi/oceanforecast/2.0/).
No API key required. Sets `User-Agent: geobn/0.1` as required by MET Norway terms of service.

Supported variables include: `sea_surface_wave_height`, `sea_water_speed`, `sea_water_temperature`.

**Example:**

```python
source = geobn.METOceanForecastSource("sea_surface_wave_height", sample_points=5)
bn.set_input("wave_height", source)
```

---

## METLocationForecastSource

::: geobn.METLocationForecastSource
    options:
      show_root_heading: true

Uses the [MET Norway LocationForecast 2.0 API](https://api.met.no/weatherapi/locationforecast/2.0/).
No API key required. Sets `User-Agent: geobn/0.1`.

Supported variables include: `wind_speed`, `wind_from_direction`, `air_temperature`.

**Example:**

```python
source = geobn.METLocationForecastSource("wind_speed", sample_points=5)
bn.set_input("wind_speed", source)
```
