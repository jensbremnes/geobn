"""Lyngen Alps avalanche risk — geobn demo.

Demonstrates pixel-wise Bayesian risk inference over real Norwegian terrain
data. The Kartverket Digital Terrain Model (10 m resolution) is fetched via a
free WCS endpoint; slope angle, aspect, and forest cover are derived
analytically from the elevation grid. Weather inputs (recent snowfall, air
temperature, wind speed) are configurable scalar constants — edit the lines at
the top of this file to explore different weather scenarios.

Data sources
------------
WCSSource (Kartverket DTM)
    Norwegian 10 m Digital Terrain Model from Kartverket's free WCS.
    Requires internet on first run; coverage: mainland Norway only.
    https://hoydedata.no/arcgis/services/las_dtm_somlos/ImageServer/WCSServer

ConstantSource
    Broadcasts a single scalar value over the entire grid.

Derived inputs
--------------
``slope_angle``   — slope in degrees computed from the DEM via numpy.gradient.
``sun_exposure``  — aspect quadrant (0=north, 1=east, 2=west, 3=south) derived
                    from the same DEM. Risk order: north > east > west > south.
``forest_cover``  — treeline heuristic: dense below 400 m, moderate 400–800 m,
                    sparse above 800 m (alpine zone). Derived from the DEM.

Bayesian network (avalanche_risk.bif)
--------------------------------------
    slope_angle ──┐
                   ├──► terrain_factor ──┐
    sun_exposure ──┤                     │
    forest_cover ──┘                     ├──► avalanche_risk
    wind_load ──┐                        │
                 ├──► weather_factor ────┘
    recent_snow ─┤
    temperature ─┘

Outputs (examples/lyngen_alps/output/)
---------------------------------------
    map.html            — interactive Leaflet map (pan/zoom, layer switcher)
                          (requires folium: pip install geobn[viz])
    avalanche_risk.tif  — 3-band GeoTIFF: P(low), P(high), entropy
                          (requires rasterio: pip install geobn[io])

Run
---
    uv run python examples/lyngen_alps/run_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import geobn

# ---------------------------------------------------------------------------
# Study area — Lyngen Alps, Tromsø county, northern Norway
# ---------------------------------------------------------------------------
WEST, SOUTH, EAST, NORTH = 19.8, 69.35, 21.0, 69.75
CRS = "EPSG:4326"
RESOLUTION = 0.005   # ~200 m at 70°N  →  80 rows × 240 cols

# ---------------------------------------------------------------------------
# Weather scenario  (edit these lines to explore different conditions)
# ---------------------------------------------------------------------------
RECENT_SNOW_CM = 30.0   # cm  — heavy recent snowfall (typical Lyngen winter)
AIR_TEMP_C     = -5.0   # °C  — cold but not extreme
WIND_SPEED_MS  =  8.0   # m/s — moderate wind loading

OUT_DIR = Path(__file__).parent / "output"
CACHE_DIR = Path(__file__).parent / "cache"  # terrain cached here after first run


# ---------------------------------------------------------------------------
# Terrain derivation from DEM
# ---------------------------------------------------------------------------

def compute_slope_aspect(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (slope_deg, sun_exposure) derived from a geographic-CRS DEM.

    Parameters
    ----------
    dem:
        Elevation array (H, W) in metres, geographic CRS EPSG:4326.
        NaN encodes nodata (sea / outside coverage).

    Returns
    -------
    slope_deg : float32 (H, W)
        Slope in degrees (0–90). NaN where DEM is NaN.
    sun_exposure : float32 (H, W)
        Aspect class as a numeric code mapped to the BN ``sun_exposure`` states:
          0 = north (315°–45°)  — highest avalanche risk
          1 = east  (45°–135°)  — second-highest risk
          2 = west  (225°–315°) — third
          3 = south (135°–225°) — lowest risk (most sun exposure)
        NaN where DEM is NaN.
    """
    lat_mid = (SOUTH + NORTH) / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_mid))
    pixel_lat_m = RESOLUTION * m_per_deg_lat   # row spacing in metres (~556 m)
    pixel_lon_m = RESOLUTION * m_per_deg_lon   # col spacing in metres (~201 m)

    # Fill NaN with 0 so gradient doesn't propagate NaN into neighbours.
    dem_filled = np.where(np.isnan(dem), 0.0, dem)

    # np.gradient(arr, dy, dx) returns (dz/dy, dz/dx).
    # Rows increase southward in a north-up raster, so dz_drow is the
    # southward partial derivative.
    dz_drow, dz_dcol = np.gradient(dem_filled, pixel_lat_m, pixel_lon_m)

    # Slope magnitude in degrees.
    slope_deg = np.degrees(
        np.arctan(np.sqrt(dz_dcol**2 + dz_drow**2))
    ).astype(np.float32)

    # Aspect as compass bearing of steepest ascent (0°=N, 90°=E, 180°=S, 270°=W).
    # East component = dz_dcol; north component = -dz_drow (rows↑ = south↓).
    aspect_compass = np.degrees(np.arctan2(dz_dcol, -dz_drow)) % 360.0

    # Classify into 4 cardinal quadrants ordered by avalanche risk (N highest, S lowest).
    sun_exposure = np.where(
        (aspect_compass >= 315.0) | (aspect_compass < 45.0), 0.0,   # north
        np.where(
            aspect_compass < 135.0, 1.0,                             # east
            np.where(
                aspect_compass < 225.0, 3.0,                         # south
                2.0,                                                  # west
            ),
        ),
    ).astype(np.float32)

    # Restore NaN mask from the original DEM.
    nodata = np.isnan(dem)
    slope_deg[nodata]    = np.nan
    sun_exposure[nodata] = np.nan

    return slope_deg, sun_exposure


def derive_forest_cover(dem: np.ndarray) -> np.ndarray:
    """Return a forest cover array derived from elevation (treeline heuristic).

    Lyngen Alps treeline is approximately 400 m. Above 800 m the terrain is
    fully alpine and offers almost no snow anchoring.

    Parameters
    ----------
    dem:
        Elevation array (H, W) in metres. NaN = nodata.

    Returns
    -------
    forest_cover : float32 (H, W)
        Numeric codes matching the BN ``forest_cover`` states:
          0 = sparse   (> 800 m — alpine zone)
          1 = moderate (400–800 m — sub-alpine)
          2 = dense    (< 400 m — forested valley)
        NaN where DEM is NaN.
    """
    forest_cover = np.where(
        dem < 400, 2.0,                 # dense forest in valley
        np.where(dem < 800, 1.0, 0.0)  # moderate sub-alpine / sparse alpine
    ).astype(np.float32)

    forest_cover[np.isnan(dem)] = np.nan
    return forest_cover


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    H = round((NORTH - SOUTH) / RESOLUTION)   # 80 rows
    W = round((EAST  - WEST)  / RESOLUTION)   # 240 cols

    print("Lyngen Alps Avalanche Risk — geobn demo")
    print(f"Study area  : {WEST}°E – {EAST}°E, {SOUTH}°N – {NORTH}°N")
    print(f"Grid        : {H} × {W} pixels at {RESOLUTION}° (~200 m)")

    # ── 1. Load BN and configure grid ─────────────────────────────────────
    bif_path = Path(__file__).parent / "avalanche_risk.bif"
    bn = geobn.load(bif_path)
    bn.set_grid(CRS, RESOLUTION, (WEST, SOUTH, EAST, NORTH))

    # ── 2. Fetch DTM and derive terrain inputs ────────────────────────────
    print("\nFetching Kartverket DTM (cached after first run) ...")
    _KARTVERKET_URL = (
        "https://hoydedata.no/arcgis/services/las_dtm_somlos/ImageServer/WCSServer"
    )
    try:
        dem = bn.fetch_raw(geobn.WCSSource(
            url=_KARTVERKET_URL,
            layer="las_dtm",
            version="1.0.0",
            format="GeoTIFF",
            valid_range=(-500.0, 9000.0),
            cache_dir=CACHE_DIR,
        ))
    except Exception as exc:
        sys.exit(f"ERROR fetching DTM: {exc}")

    dem[dem <= 0] = np.nan   # ocean / fjord surfaces (Kartverket returns 0 for sea level)
    slope_deg, sun_exposure = compute_slope_aspect(dem)
    forest_cover = derive_forest_cover(dem)

    land_pixels = int(np.isfinite(dem).sum())
    north_pct = 100.0 * float(np.nanmean(sun_exposure == 0.0))
    print(f"Terrain     : {land_pixels:,} land pixels  (N-facing: {north_pct:.1f}%)")
    print(f"Slope range : {np.nanmin(slope_deg):.1f}° – "
          f"{np.nanmax(slope_deg):.1f}°  (mean: {np.nanmean(slope_deg):.1f}°)")

    dense_pct    = 100.0 * float(np.nanmean(forest_cover == 2.0))
    moderate_pct = 100.0 * float(np.nanmean(forest_cover == 1.0))
    sparse_pct   = 100.0 * float(np.nanmean(forest_cover == 0.0))
    print(f"Forest cover: dense {dense_pct:.0f}%  moderate {moderate_pct:.0f}%  sparse {sparse_pct:.0f}%")

    # ── 3. Wire inputs ─────────────────────────────────────────────────────
    bn.set_input("slope_angle",  geobn.ArraySource(slope_deg))
    bn.set_input("sun_exposure", geobn.ArraySource(sun_exposure))
    bn.set_input("forest_cover", geobn.ArraySource(forest_cover))
    bn.set_input("recent_snow", geobn.ConstantSource(RECENT_SNOW_CM))
    bn.set_input("temperature",  geobn.ConstantSource(AIR_TEMP_C))
    bn.set_input("wind_load",    geobn.ConstantSource(WIND_SPEED_MS))

    # ── 4. Discretizations ────────────────────────────────────────────────
    bn.set_discretization("slope_angle",  [0, 5, 25, 40, 90])
    bn.set_discretization("sun_exposure", [-0.5, 0.5, 1.5, 2.5, 3.5])
    bn.set_discretization("forest_cover", [-0.5, 0.5, 1.5, 2.5])   # sparse / moderate / dense
    bn.set_discretization("recent_snow",  [0, 15, 35, 150])
    bn.set_discretization("temperature",  [-40, -8, -2, 15])
    bn.set_discretization("wind_load",    [0, 5, 15, 50])            # low / moderate / high (m/s)

    # ── 5. Weather scenario summary ────────────────────────────────────────
    snow_state = (
        "light"    if RECENT_SNOW_CM < 10
        else "moderate" if RECENT_SNOW_CM < 25
        else "heavy"
    )
    temp_state = (
        "cold"     if AIR_TEMP_C < -8
        else "moderate" if AIR_TEMP_C < -2
        else "warming"
    )
    wind_state = (
        "low"      if WIND_SPEED_MS < 5
        else "moderate" if WIND_SPEED_MS < 15
        else "high"
    )
    print(f"\nWeather scenario")
    print(f"  Recent snow  : {RECENT_SNOW_CM:.0f} cm   → {snow_state}")
    print(f"  Temperature  : {AIR_TEMP_C:.0f}°C   → {temp_state}")
    print(f"  Wind speed   : {WIND_SPEED_MS:.0f} m/s  → {wind_state}")

    # ── 6. Run inference ───────────────────────────────────────────────────
    print("\nRunning BN inference ...")
    try:
        result = bn.infer(query=["avalanche_risk"])
    except Exception as exc:
        sys.exit(f"ERROR during inference: {exc}")

    probs = result.probabilities["avalanche_risk"]   # (H, W, 2)
    ent   = result.entropy("avalanche_risk")          # (H, W)

    # ── 7. Console statistics ──────────────────────────────────────────────
    def bar(val: float, width: int = 20) -> str:
        filled = round(val * width)
        return "█" * filled + "░" * (width - filled)

    print("\n── Avalanche risk distribution ──────────────────────────────────")
    for i, state in enumerate(result.state_names["avalanche_risk"]):
        p = float(np.nanmean(probs[..., i]))
        print(f"  P({state:6s}) mean {p:.2f}  {bar(p)}")

    p_high = probs[..., 1]
    steep_north  = (slope_deg > 35) & (sun_exposure == 0.0)   # north-facing
    gentle_south = (slope_deg < 25) & (sun_exposure == 3.0)   # south-facing
    p_high_steep_north  = float(np.nanmean(p_high[steep_north]))  if steep_north.any()  else float("nan")
    p_high_gentle_south = float(np.nanmean(p_high[gentle_south])) if gentle_south.any() else float("nan")

    print("\n── Risk by terrain type ─────────────────────────────────────────")
    print(f"  Steep N-facing slopes (>35°, N-facing)  : P(high) = {p_high_steep_north:.2f}")
    print(f"  Gentle S-facing slopes (<25°, S-facing) : P(high) = {p_high_gentle_south:.2f}")

    # ── 8. Interactive map ─────────────────────────────────────────────────
    try:
        html_path = result.show_map(
            OUT_DIR,
            extra_layers={
                "Slope angle (°)": slope_deg,
                "Sun exposure": sun_exposure,
                "Forest cover": forest_cover,
            },
        )
        print(f"\nInteractive map opened in browser → {html_path}")
        print("  Use the layer control (top-right) to switch overlays.")
    except ImportError as exc:
        print(f"\nSkipping interactive map ({exc})")
        print("  Install folium: pip install geobn[viz]")

    # ── 9. Export GeoTIFF ─────────────────────────────────────────────────
    try:
        result.to_geotiff(OUT_DIR)
        tif_path = OUT_DIR / "avalanche_risk.tif"
        print(f"GeoTIFF written → {tif_path}")
        print("  Band 1: P(low)   Band 2: P(high)   Band 3: entropy")
    except ImportError as exc:
        print(f"\nSkipping GeoTIFF export ({exc})")
        print("  Install rasterio: pip install geobn[io]")


if __name__ == "__main__":
    main()
