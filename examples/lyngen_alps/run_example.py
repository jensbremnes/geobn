"""Lyngen Alps avalanche risk — geobn demo.

Demonstrates pixel-wise Bayesian risk inference over real Norwegian terrain
data. The Kartverket Digital Terrain Model (10 m resolution) is fetched via a
free WCS endpoint; slope angle, aspect, and forest cover are derived
from the elevation grid with ``geobn.terrain`` and ``geobn.DerivedSource``. Weather inputs (recent snowfall, air
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
``slope_angle``   — ``geobn.terrain.slope``: slope in degrees from the DEM
                    (one-sided differences next to sea, so no fake coastal cliffs).
``sun_exposure``  — quadrant of ``geobn.terrain.aspect`` (0=north, 1=east,
                    2=west, 3=south). Risk order: north > east > west > south.
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
    avalanche_risk.tif  — 3-band GeoTIFF: P(low), P(high), entropy

Run
---
    uv run python examples/lyngen_alps/run_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import geobn

# This script prints arrows and box-drawing rules, which the default console
# encoding on Windows (cp1252) cannot represent.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
# Terrain classes derived from the DEM
# ---------------------------------------------------------------------------

def mask_sea(dem: np.ndarray) -> np.ndarray:
    """Kartverket returns 0 for sea and fjord surfaces; treat them as nodata."""
    return np.where(dem > 0, dem, np.nan)


def aspect_quadrant(aspect_deg: np.ndarray) -> np.ndarray:
    """Classify aspect (degrees from north) into the BN ``sun_exposure`` codes.

    The north quadrant wraps around 0°, so this cannot be expressed as
    breakpoints and is done here instead:
      0 = north (315°–45°)  — highest avalanche risk
      1 = east  (45°–135°)  — second-highest risk
      2 = west  (225°–315°) — third
      3 = south (135°–225°) — lowest risk (most sun exposure)
    NaN where the aspect is NaN, including perfectly flat cells.
    """
    quadrant = np.where(
        (aspect_deg >= 315.0) | (aspect_deg < 45.0), 0.0,    # north
        np.where(
            aspect_deg < 135.0, 1.0,                          # east
            np.where(aspect_deg < 225.0, 3.0, 2.0),           # south / west
        ),
    )
    return np.where(np.isnan(aspect_deg), np.nan, quadrant)


def treeline_forest_cover(dem: np.ndarray) -> np.ndarray:
    """Forest cover from elevation (treeline heuristic).

    Lyngen Alps treeline is approximately 400 m. Above 800 m the terrain is
    fully alpine and offers almost no snow anchoring. Codes match the BN
    ``forest_cover`` states:
      0 = sparse   (> 800 m — alpine zone)
      1 = moderate (400–800 m — sub-alpine)
      2 = dense    (< 400 m — forested valley)
    NaN where the DEM is NaN.
    """
    cover = np.where(dem < 400, 2.0, np.where(dem < 800, 1.0, 0.0))
    return np.where(np.isnan(dem), np.nan, cover)


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

    # ── 2. Terrain inputs derived from the DTM ────────────────────────────
    # Nothing is fetched here: the sources describe how each input is made,
    # and the DTM is downloaded when the first of them is needed (then read
    # from the disk cache).
    dem = geobn.DerivedSource(mask_sea, geobn.WCSSource(
        url="https://hoydedata.no/arcgis/services/las_dtm_somlos/ImageServer/WCSServer",
        layer="las_dtm",
        version="1.0.0",
        format="GeoTIFF",
        valid_range=(-500.0, 9000.0),
        cache_dir=CACHE_DIR,
    ))
    slope = geobn.terrain.slope(dem)
    sun_exposure = geobn.DerivedSource(aspect_quadrant, geobn.terrain.aspect(dem))
    forest_cover = geobn.DerivedSource(treeline_forest_cover, dem)

    # ── 3. Wire inputs ─────────────────────────────────────────────────────
    bn.set_input("slope_angle",  slope)
    bn.set_input("sun_exposure", sun_exposure)
    bn.set_input("forest_cover", forest_cover)
    bn.set_input("recent_snow", geobn.ConstantSource(RECENT_SNOW_CM))
    bn.set_input("temperature",  geobn.ConstantSource(AIR_TEMP_C))
    bn.set_input("wind_load",    geobn.ConstantSource(WIND_SPEED_MS))

    # The terrain arrays themselves, for the summary below and the map layers.
    print("\nFetching Kartverket DTM (cached after first run) ...")
    try:
        land_pixels = int(np.isfinite(bn.fetch_raw(dem)).sum())
    except Exception as exc:
        sys.exit(f"ERROR fetching DTM: {exc}")
    slope_deg = bn.fetch_raw(slope)
    exposure = bn.fetch_raw(sun_exposure)
    cover = bn.fetch_raw(forest_cover)

    north_pct = 100.0 * float(np.nanmean(exposure == 0.0))
    print(f"Terrain     : {land_pixels:,} land pixels  (N-facing: {north_pct:.1f}%)")
    print(f"Slope range : {np.nanmin(slope_deg):.1f}° – "
          f"{np.nanmax(slope_deg):.1f}°  (mean: {np.nanmean(slope_deg):.1f}°)")

    dense_pct    = 100.0 * float(np.nanmean(cover == 2.0))
    moderate_pct = 100.0 * float(np.nanmean(cover == 1.0))
    sparse_pct   = 100.0 * float(np.nanmean(cover == 0.0))
    print(f"Forest cover: dense {dense_pct:.0f}%  moderate {moderate_pct:.0f}%  sparse {sparse_pct:.0f}%")

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
    print("\nWeather scenario")
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

    # ── 7. Console statistics ──────────────────────────────────────────────
    def bar(val: float, width: int = 20) -> str:
        filled = round(val * width)
        return "█" * filled + "░" * (width - filled)

    print("\n── Avalanche risk distribution ──────────────────────────────────")
    for i, state in enumerate(result.state_names["avalanche_risk"]):
        p = float(np.nanmean(probs[..., i]))
        print(f"  P({state:6s}) mean {p:.2f}  {bar(p)}")

    p_high = probs[..., 1]
    steep_north  = (slope_deg > 35) & (exposure == 0.0)   # north-facing
    gentle_south = (slope_deg < 25) & (exposure == 3.0)   # south-facing
    p_high_steep_north  = float(np.nanmean(p_high[steep_north]))  if steep_north.any()  else float("nan")
    p_high_gentle_south = float(np.nanmean(p_high[gentle_south])) if gentle_south.any() else float("nan")

    print("\n── Risk by terrain type ─────────────────────────────────────────")
    print(f"  Steep N-facing slopes (>35°, N-facing)  : P(high) = {p_high_steep_north:.2f}")
    print(f"  Gentle S-facing slopes (<25°, S-facing) : P(high) = {p_high_gentle_south:.2f}")

    # ── 8. Interactive map ─────────────────────────────────────────────────
    html_path = result.show_map(
        OUT_DIR,
        extra_layers={
            "Slope angle (°)": slope_deg,
            "Sun exposure": exposure,
            "Forest cover": cover,
        },
    )
    print(f"\nInteractive map opened in browser → {html_path}")
    print("  Use the layer control (top-right) to switch overlays.")

    # ── 9. Export GeoTIFF ─────────────────────────────────────────────────
    result.to_geotiff(OUT_DIR)
    tif_path = OUT_DIR / "avalanche_risk.tif"
    print(f"GeoTIFF written → {tif_path}")
    print("  Band 1: P(low)   Band 2: P(high)   Band 3: entropy")


if __name__ == "__main__":
    main()
