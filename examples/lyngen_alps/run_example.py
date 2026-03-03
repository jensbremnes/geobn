"""Lyngen Alps avalanche risk — geobn demo.

Demonstrates pixel-wise Bayesian risk inference over real Norwegian terrain
data. The Kartverket Digital Terrain Model (10 m resolution) is fetched via a
free WCS endpoint; slope angle and aspect are derived analytically from the
elevation grid. Weather inputs (recent snowfall, air temperature) are
configurable scalar constants — edit the two lines at the top of this file to
explore different weather scenarios.

Data sources
------------
KartverketDTMSource
    Norwegian 10 m Digital Terrain Model from Kartverket's free WCS.
    Requires internet on first run; coverage: mainland Norway only.
    https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm10

ConstantSource
    Broadcasts a single scalar value over the entire grid.

Derived inputs
--------------
``slope_angle``  — slope in degrees computed from the DEM via numpy.gradient.
``aspect``       — binary N-facing indicator (0 = south-facing = favorable,
                   1 = north-facing = unfavorable) derived from the same DEM.

Bayesian network (avalanche_risk.bif)
--------------------------------------
    slope_angle ──┐
                   ├──► terrain_factor ──┐
    aspect ────────┘                     ├──► avalanche_risk
    recent_snow ──┐                      │
                   ├──► weather_factor ──┘
    temperature ──┘

Outputs (examples/lyngen_alps/output/)
---------------------------------------
    map.html            — interactive Leaflet map (pan/zoom, layer switcher)
                          (requires folium: pip install geobn[viz])
    avalanche_risk.tif  — 4-band GeoTIFF: P(low), P(medium), P(high), entropy
                          (requires rasterio: pip install geobn[io])

Run
---
    uv run python examples/lyngen_alps/run_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from affine import Affine

import geobn
from geobn.grid import GridSpec

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


# ---------------------------------------------------------------------------
# Slope and aspect derivation
# ---------------------------------------------------------------------------

def compute_slope_aspect(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (slope_deg, north_facing) derived from a geographic-CRS DEM.

    Parameters
    ----------
    dem:
        Elevation array (H, W) in metres, geographic CRS EPSG:4326.
        NaN encodes nodata (sea / outside coverage).

    Returns
    -------
    slope_deg : float32 (H, W)
        Slope in degrees (0–90). NaN where DEM is NaN.
    north_facing : float32 (H, W)
        1.0 where aspect is N-facing (NW–NE, compass bearing 270°–360° or
        0°–90°); 0.0 where S-facing. NaN where DEM is NaN.
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

    # North-facing: bearings 270°–360° (NW) and 0°–90° (NE) are unfavorable.
    north_facing = (
        (aspect_compass >= 270.0) | (aspect_compass < 90.0)
    ).astype(np.float32)

    # Restore NaN mask from the original DEM.
    nodata = np.isnan(dem)
    slope_deg[nodata]   = np.nan
    north_facing[nodata] = np.nan

    return slope_deg, north_facing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    H = round((NORTH - SOUTH) / RESOLUTION)   # 80 rows
    W = round((EAST  - WEST)  / RESOLUTION)   # 240 cols
    transform = Affine(RESOLUTION, 0, WEST, 0, -RESOLUTION, NORTH)
    ref_grid = GridSpec(crs=CRS, transform=transform, shape=(H, W))

    print("Lyngen Alps Avalanche Risk — geobn demo")
    print(f"Study area  : {WEST}°E – {EAST}°E, {SOUTH}°N – {NORTH}°N")
    print(f"Grid        : {H} × {W} pixels at {RESOLUTION}° (~200 m)")

    # ── 1. Fetch Kartverket DTM ────────────────────────────────────────────
    print("\nFetching Kartverket DTM (requires internet) ...")
    try:
        dtm_data = geobn.KartverketDTMSource().fetch(grid=ref_grid)
    except Exception as exc:
        sys.exit(f"ERROR fetching DTM: {exc}")

    dem = dtm_data.array

    # Derive slope and aspect from the DEM.
    slope_deg, north_facing = compute_slope_aspect(dem)

    land_pixels = int(np.isfinite(dem).sum())
    n_facing_pct = 100.0 * float(np.nanmean(north_facing))
    print(f"Terrain     : {land_pixels:,} land pixels  (N-facing: {n_facing_pct:.1f}%)")
    print(f"Slope range : {np.nanmin(slope_deg):.1f}° – "
          f"{np.nanmax(slope_deg):.1f}°  (mean: {np.nanmean(slope_deg):.1f}°)")

    # ── 2. Load BN and wire inputs ─────────────────────────────────────────
    bif_path = Path(__file__).parent / "avalanche_risk.bif"
    bn = geobn.load(bif_path)
    bn.set_grid(CRS, RESOLUTION, (WEST, SOUTH, EAST, NORTH))

    bn.set_input(
        "slope_angle",
        geobn.ArraySource(slope_deg, crs=CRS, transform=dtm_data.transform),
    )
    bn.set_input(
        "aspect",
        geobn.ArraySource(north_facing, crs=CRS, transform=dtm_data.transform),
    )
    bn.set_input("recent_snow", geobn.ConstantSource(RECENT_SNOW_CM))
    bn.set_input("temperature", geobn.ConstantSource(AIR_TEMP_C))

    # ── 3. Discretizations ────────────────────────────────────────────────
    bn.set_discretization("slope_angle", [0, 25, 40, 90],   ["gentle", "steep", "extreme"])
    bn.set_discretization("aspect",      [0.0, 0.5, 1.5],   ["favorable", "unfavorable"])
    bn.set_discretization("recent_snow", [0, 10, 25, 150],  ["light", "moderate", "heavy"])
    bn.set_discretization("temperature", [-40, -8, -2, 15], ["cold", "moderate", "warming"])

    # ── 4. Weather scenario summary ────────────────────────────────────────
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

    # ── 5. Run inference ───────────────────────────────────────────────────
    print("\nRunning BN inference ...")
    try:
        result = bn.infer(query=["avalanche_risk"])
    except Exception as exc:
        sys.exit(f"ERROR during inference: {exc}")

    probs = result.probabilities["avalanche_risk"]   # (H, W, 3)
    ent   = result.entropy("avalanche_risk")          # (H, W)

    # ── 6. Console statistics ──────────────────────────────────────────────
    def bar(val: float, width: int = 20) -> str:
        filled = round(val * width)
        return "█" * filled + "░" * (width - filled)

    print("\n── Avalanche risk distribution ──────────────────────────────────")
    for i, state in enumerate(result.state_names["avalanche_risk"]):
        p = float(np.nanmean(probs[..., i]))
        print(f"  P({state:6s}) mean {p:.2f}  {bar(p)}")

    p_high = probs[..., 2]
    steep_n  = (slope_deg > 35) & (north_facing == 1.0)
    gentle_s = (slope_deg < 25) & (north_facing == 0.0)
    p_high_steep_n  = float(np.nanmean(p_high[steep_n]))  if steep_n.any()  else float("nan")
    p_high_gentle_s = float(np.nanmean(p_high[gentle_s])) if gentle_s.any() else float("nan")

    print("\n── Risk by terrain type ─────────────────────────────────────────")
    print(f"  Steep N-facing slopes (>35°, N-facing)  : P(high) = {p_high_steep_n:.2f}")
    print(f"  Gentle S-facing slopes (<25°, S-facing) : P(high) = {p_high_gentle_s:.2f}")

    # ── 7. Interactive map ─────────────────────────────────────────────────
    try:
        html_path = result.show_map(
            OUT_DIR,
            extra_layers={"Slope angle (°)": slope_deg},
        )
        print(f"\nInteractive map opened in browser → {html_path}")
        print("  Use the layer control (top-right) to switch overlays.")
    except ImportError as exc:
        print(f"\nSkipping interactive map ({exc})")
        print("  Install folium: pip install geobn[viz]")

    # ── 8. Export GeoTIFF ─────────────────────────────────────────────────
    try:
        result.to_geotiff(OUT_DIR)
        tif_path = OUT_DIR / "avalanche_risk.tif"
        print(f"GeoTIFF written → {tif_path}")
        print("  Band 1: P(low)   Band 2: P(medium)   Band 3: P(high)   Band 4: entropy")
    except ImportError as exc:
        print(f"\nSkipping GeoTIFF export ({exc})")
        print("  Install rasterio: pip install geobn[io]")


if __name__ == "__main__":
    main()
