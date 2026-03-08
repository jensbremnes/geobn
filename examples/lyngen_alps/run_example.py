"""Lyngen Alps avalanche risk — geobn demo.

Demonstrates pixel-wise Bayesian risk inference over real Norwegian terrain
data. The Kartverket Digital Terrain Model (10 m resolution) is fetched via a
free WCS endpoint; slope angle and aspect are derived analytically from the
elevation grid. Weather inputs (recent snowfall, air temperature) are
configurable scalar constants — edit the two lines at the top of this file to
explore different weather scenarios.

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
``slope_angle``  — slope in degrees computed from the DEM via numpy.gradient.
``sun_exposure``  — aspect quadrant (0=north, 1=east, 2=west, 3=south) derived
                   from the same DEM. Risk order: north > east > west > south.

Bayesian network (avalanche_risk.bif)
--------------------------------------
    slope_angle ──┐
                   ├──► terrain_factor ──┐
    sun_exposure ──┘                     ├──► avalanche_risk
    recent_snow ──┐                      │
                   ├──► weather_factor ──┘
    temperature ──┘

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
# Weather scenario  (edit these two lines to explore different conditions)
# ---------------------------------------------------------------------------
RECENT_SNOW_CM = 30.0   # cm  — heavy recent snowfall (typical Lyngen winter)
AIR_TEMP_C     = -5.0   # °C  — cold but not extreme

OUT_DIR = Path(__file__).parent / "output"
CACHE_DIR = Path(__file__).parent / "cache"  # terrain cached here after first run


# ---------------------------------------------------------------------------
# Slope and aspect derivation
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

    land_pixels = int(np.isfinite(dem).sum())
    north_pct = 100.0 * float(np.nanmean(sun_exposure == 0.0))
    print(f"Terrain     : {land_pixels:,} land pixels  (N-facing: {north_pct:.1f}%)")
    print(f"Slope range : {np.nanmin(slope_deg):.1f}° – "
          f"{np.nanmax(slope_deg):.1f}°  (mean: {np.nanmean(slope_deg):.1f}°)")

    # ── 3. Wire inputs ─────────────────────────────────────────────────────
    bn.set_input_array("slope_angle",  slope_deg)
    bn.set_input_array("sun_exposure", sun_exposure)
    bn.set_input("recent_snow", geobn.ConstantSource(RECENT_SNOW_CM))
    bn.set_input("temperature",  geobn.ConstantSource(AIR_TEMP_C))

    # ── 4. Discretizations ────────────────────────────────────────────────
    bn.set_discretization("slope_angle", [0, 5, 25, 40, 90])
    bn.set_discretization("sun_exposure", [-0.5, 0.5, 1.5, 2.5, 3.5])
    bn.set_discretization("recent_snow", [0, 10, 25, 150])
    bn.set_discretization("temperature", [-40, -8, -2, 15])

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
    print(f"\nWeather scenario")
    print(f"  Recent snow  : {RECENT_SNOW_CM:.0f} cm   → {snow_state}")
    print(f"  Temperature  : {AIR_TEMP_C:.0f}°C   → {temp_state}")

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
