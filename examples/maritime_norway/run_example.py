"""Maritime navigation risk — Norwegian outer coast (Stad–Ålesund).

Demonstrates pixel-wise Bayesian risk inference from exclusively Norwegian
and European open data (all free, no credentials required).

Data sources
------------
EMODnet Bathymetry
    European seabed depth grid (~250 m resolution), from
    https://ows.emodnet-bathymetry.eu/wcs — no auth, no registration.

EMODnet Human Activities
    Historical vessel traffic density (vessel hours/km²/month), derived
    from satellite AIS data, from
    https://www.emodnet-humanactivities.eu/geoserver/emodnet/wcs
    — no auth, no registration.

MET Norway OceanForecast 2.0
    Wave height and sea current speed, sampled on a 5×5 point grid over
    the study area.  https://api.met.no/weatherapi/oceanforecast/2.0/

MET Norway LocationForecast 2.0
    Wind speed and wind direction, sampled on a 5×5 / 3×3 point grid.
    https://api.met.no/weatherapi/locationforecast/2.0/

Derived inputs
--------------
``water_depth``   — positive depth (m), from EMODnet Bathymetry (sign flip);
                    NaN on land and intertidal areas.
``wind_onshore``  — binary spatial indicator: 1.0 where wind blows toward
                    shallower water (increasing grounding risk), 0.0 elsewhere.
                    Computed from mean wind direction × bathymetry gradient.

Bayesian network (maritime_norway.bif)
--------------------------------------
    wave_height  ─┐
                  ├─► sea_state ─┬─► grounding_risk
    current_speed ┘              │         ↑
                                 └─► collision_risk
                                           ↑
    water_depth  ────────────────── grounding_risk
    wind_onshore ────────────────── grounding_risk
    vessel_density ──────────────── collision_risk
    wind_speed   ────────────────── collision_risk

Outputs (examples/maritime_norway/output/)
------------------------------------------
    grounding_risk.tif — 4 bands: P(low), P(medium), P(high), entropy
    collision_risk.tif — 4 bands: P(low), P(medium), P(high), entropy

    Land pixels are NaN in all outputs (ships don't navigate on land).

Run
---
    uv run python examples/maritime_norway/run_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from affine import Affine

import geobn
from geobn.grid import GridSpec

# ---------------------------------------------------------------------------
# Study area — outer coast between Stad and Ålesund, western Norway
# ---------------------------------------------------------------------------
WEST, SOUTH, EAST, NORTH = 4.8, 62.0, 6.5, 63.0
CRS = "EPSG:4326"
RESOLUTION = 0.01  # degrees (~1 km at this latitude)

OUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Wind-onshore derivation
# ---------------------------------------------------------------------------

def compute_wind_onshore(depth_m: np.ndarray, mean_wind_from_deg: float) -> np.ndarray:
    """Return a binary (0.0/1.0) array indicating where wind blows toward shallower water.

    Parameters
    ----------
    depth_m:
        Positive water depth array (m); NaN for land.  Shape (H, W).
    mean_wind_from_deg:
        Single representative wind direction for the study area, in
        meteorological convention: degrees FROM which wind blows,
        measured clockwise from north (0°=north, 90°=east, 270°=west).

    Returns
    -------
    np.ndarray of float32, shape (H, W)
        1.0 where the wind is directed toward shallower water (onshore),
        0.0 elsewhere (offshore).  NaN pixels follow ``depth_m``.
    """
    # Fill NaN (land) with 0 so gradient doesn't propagate NaN into the sea
    depth_filled = np.where(np.isnan(depth_m), 0.0, depth_m)

    # Gradient in row/column (image) space:
    #   dy: gradient along rows (+row = southward in geographic space)
    #   dx: gradient along cols (+col = eastward)
    dy, dx = np.gradient(depth_filled)

    # Direction that points toward decreasing depth (toward land / shallower water).
    # arctan2(-dy, dx): in geographic space, -dy points north where depth decreases
    # going south; dx is east.  The result is in standard math convention
    # (radians, 0=east, π/2=north, CCW).
    toward_shallow_rad = np.arctan2(-dy, dx)

    # Convert meteorological wind "from" direction to math convention (CCW from east).
    # Wind FROM 270° (west) blows TOWARD east (0° math = 0° from east).
    wind_to_deg_math = 90.0 - ((mean_wind_from_deg + 180.0) % 360.0)
    wind_to_rad = np.radians(wind_to_deg_math)

    # Absolute angular difference between wind direction and toward-shallow direction.
    # Values < π/2 mean wind has a component toward shallower water.
    angle_diff = np.abs(
        ((wind_to_rad - toward_shallow_rad + np.pi) % (2.0 * np.pi)) - np.pi
    )
    wind_onshore = (angle_diff < (np.pi / 2.0)).astype(np.float32)

    # Preserve NaN mask from depth
    wind_onshore[np.isnan(depth_m)] = np.nan

    return wind_onshore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Reference grid ─────────────────────────────────────────────────
    transform = Affine(RESOLUTION, 0, WEST, 0, -RESOLUTION, NORTH)
    H = round((NORTH - SOUTH) / RESOLUTION)   # 100 rows
    W = round((EAST - WEST) / RESOLUTION)     # 170 cols
    ref_grid = GridSpec(crs=CRS, transform=transform, shape=(H, W))

    print(f"Study area      : {WEST}°E – {EAST}°E, {SOUTH}°N – {NORTH}°N")
    print(f"Grid            : {H} × {W} pixels at {RESOLUTION}° (~1 km) resolution")

    # ── 2. Bathymetry → water depth + land mask ────────────────────────────
    print("\nFetching EMODnet Bathymetry ...")
    try:
        bathy = geobn.EMODnetBathymetrySource().fetch(grid=ref_grid)
    except Exception as exc:
        sys.exit(f"ERROR fetching bathymetry: {exc}")

    # EMODnet returns negative values for ocean depth; positive = land/intertidal
    depth_m = np.where(bathy.array < 0, -bathy.array, np.nan).astype(np.float32)
    sea_pixels = np.isfinite(depth_m).sum()
    print(f"  Sea pixels    : {sea_pixels:,} / {H * W:,}  "
          f"({100 * sea_pixels / (H * W):.1f}%)")
    print(f"  Depth range   : {np.nanmin(depth_m):.0f} – {np.nanmax(depth_m):.0f} m")

    # ── 3. Wind direction → derive wind_onshore ────────────────────────────
    print("\nFetching wind direction from MET Norway LocationForecast ...")
    try:
        wind_dir_data = geobn.METLocationForecastSource(
            "wind_from_direction", sample_points=3
        ).fetch(grid=ref_grid)
        mean_wind_from = float(np.nanmean(wind_dir_data.array))
        if np.isnan(mean_wind_from):
            raise ValueError("All wind direction values are NaN")
    except Exception as exc:
        # Fall back to prevailing westerly (North Atlantic) if API is unavailable
        mean_wind_from = 270.0
        print(f"  WARNING: {exc}")
        print(f"  Using fallback wind direction: {mean_wind_from}° (from west)")
    else:
        print(f"  Mean wind from: {mean_wind_from:.1f}°")

    wind_onshore_arr = compute_wind_onshore(depth_m, mean_wind_from)
    onshore_frac = np.nanmean(wind_onshore_arr)
    print(f"  Onshore pixels: {100 * onshore_frac:.1f}% of sea area")

    # ── 4. Load BN and wire inputs ─────────────────────────────────────────
    bif_path = Path(__file__).parent / "maritime_norway.bif"
    bn = geobn.load(bif_path)
    bn.set_grid(CRS, RESOLUTION, (WEST, SOUTH, EAST, NORTH))

    bn.set_input(
        "water_depth",
        geobn.ArraySource(depth_m, crs=CRS, transform=bathy.transform),
    )
    bn.set_input(
        "wind_onshore",
        geobn.ArraySource(wind_onshore_arr, crs=CRS, transform=bathy.transform),
    )
    bn.set_input(
        "wave_height",
        geobn.METOceanForecastSource("sea_surface_wave_height", sample_points=5),
    )
    bn.set_input(
        "current_speed",
        geobn.METOceanForecastSource("sea_water_speed", sample_points=5),
    )
    bn.set_input(
        "wind_speed",
        geobn.METLocationForecastSource("wind_speed", sample_points=5),
    )
    bn.set_input(
        "vessel_density",
        geobn.EMODnetShippingDensitySource(ship_type="all", year=2022),
    )

    # ── 5. Discretizations ────────────────────────────────────────────────
    bn.set_discretization("wave_height",    [0, 0.5, 2.0, 10.0],   ["calm", "moderate", "rough"])
    bn.set_discretization("current_speed",  [0, 0.3, 1.0, 5.0],    ["low", "moderate", "high"])
    bn.set_discretization("wind_speed",     [0, 5, 15, 50],         ["calm", "moderate", "strong"])
    bn.set_discretization("water_depth",    [0, 20, 100, 9999],     ["very_shallow", "shallow", "deep"])
    bn.set_discretization("wind_onshore",   [0.0, 0.5, 1.5],        ["offshore", "onshore"])
    bn.set_discretization("vessel_density", [0, 0.5, 5.0, 1_000_000], ["sparse", "moderate", "dense"])

    # ── 6. Inference ──────────────────────────────────────────────────────
    print("\nFetching ocean/weather forecasts and running inference ...")
    print("  (MET Norway APIs: wave height, current speed, wind speed)")
    print("  (EMODnet Human Activities: vessel traffic density)")

    try:
        result = bn.infer(query=["grounding_risk", "collision_risk"])
    except Exception as exc:
        sys.exit(f"ERROR during inference: {exc}")

    # ── 7. Print statistics ───────────────────────────────────────────────
    print("\n── Grounding risk ──────────────────────────────────────────────")
    g_probs = result.probabilities["grounding_risk"]   # (H, W, 3)
    g_ent   = result.entropy("grounding_risk")
    for i, state in enumerate(result.state_names["grounding_risk"]):
        p = g_probs[..., i]
        print(f"  P({state:6s}) — min / mean / max : "
              f"{np.nanmin(p):.3f} / {np.nanmean(p):.3f} / {np.nanmax(p):.3f}")
    print(f"  Entropy (bits)  — mean : {np.nanmean(g_ent):.3f}")

    print("\n── Collision risk ──────────────────────────────────────────────")
    c_probs = result.probabilities["collision_risk"]
    c_ent   = result.entropy("collision_risk")
    for i, state in enumerate(result.state_names["collision_risk"]):
        p = c_probs[..., i]
        print(f"  P({state:6s}) — min / mean / max : "
              f"{np.nanmin(p):.3f} / {np.nanmean(p):.3f} / {np.nanmax(p):.3f}")
    print(f"  Entropy (bits)  — mean : {np.nanmean(c_ent):.3f}")

    # ── 8. xarray summary ─────────────────────────────────────────────────
    ds = result.to_xarray()
    print(f"\nxarray Dataset:\n{ds}")

    # ── 9. Export GeoTIFFs ────────────────────────────────────────────────
    try:
        result.to_geotiff(OUT_DIR)
        print(f"\nGeoTIFFs written to {OUT_DIR}/")
        print("  grounding_risk.tif — Band 1: P(low)   Band 2: P(medium)   "
              "Band 3: P(high)   Band 4: entropy")
        print("  collision_risk.tif — Band 1: P(low)   Band 2: P(medium)   "
              "Band 3: P(high)   Band 4: entropy")
    except ImportError as exc:
        print(f"\nSkipping GeoTIFF export ({exc})")
        print("  Install rasterio: pip install geobn[io]")


if __name__ == "__main__":
    main()
