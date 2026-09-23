"""Karmsundet USV Maritime Risk — geobn demo.

Demonstrates pixel-wise Bayesian risk inference for an unmanned surface vessel
(USV) operating in the Karmsundet strait near Haugesund, Norway. Karmsundet is
a busy, narrow tidal channel between Karmøy island and the mainland, with
significant commercial and recreational vessel traffic, tidal currents, and
occasional sea-fog.

The Bayesian network combines three risk dimensions:

  water_depth ──────────────────┐
  current_speed ────────────────┼──► grounding_risk ─────────┐
                                │                             │
  vessel_traffic ───────────────┐                             │
  wave_height ──────────────────┼──► collision_risk ──────────┼──► usv_risk
  fog_fraction ─────────────────┘                             │
                                                              │
  wave_height ──────────────────┐                             │
  wind_speed ───────────────────┼──► navigation_difficulty ───┘
  current_speed ────────────────┘

Data sources
------------
water_depth (WCSSource)
    EMODnet Bathymetry WCS — free, global, ~115 m resolution.
    Negative values = below sea level; flipped to positive depth below surface.
    URL: https://ows.emodnet-bathymetry.eu/wcs

vessel_traffic (MosaicSource over RasterSource and ConstantSource)
    Pre-processed AIS density GeoTIFF (encounters/km²/day) placed at
    ``data/ais_density_karmsundet.tif``, backed by ConstantSource(2.0)
    (medium traffic) wherever the raster has no data — run
    ``create_ais_density.py`` to generate the real raster from Kystverket AIS
    data.

wave_height, current_speed (PointGridSource → Met.no Oceanforecast)
    Live ocean forecast sampled on a 5×5 grid across the study area.
    Variables: sea_surface_wave_height, sea_water_speed.

wind_speed, fog_fraction (PointGridSource → Met.no Locationforecast)
    Live atmospheric forecast on the same 5×5 grid.
    Variables: wind_speed, fog_area_fraction.

Outputs (examples/karmsundet/output/)
---------------------------------------
    usv_risk_map.html   — interactive Leaflet map with USV risk overlays
    usv_risk.tif        — 4-band GeoTIFF: P(low), P(medium), P(high), entropy
    risk_score.tif      — float32 scalar risk score 10–90

Run
---
    uv run python examples/karmsundet/run_example.py
"""
from __future__ import annotations

import json
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np

import geobn

# This script prints arrows and box-drawing rules, which the default console
# encoding on Windows (cp1252) cannot represent.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Study area — Karmsundet strait, Haugesund, western Norway
# ---------------------------------------------------------------------------
WEST, SOUTH, EAST, NORTH = 5.15, 59.25, 5.55, 59.55
CRS = "EPSG:4326"
RESOLUTION = 0.002   # ~200 m at 59°N  →  150 rows × 200 cols

HERE      = Path(__file__).parent
OUT_DIR   = HERE / "output"
CACHE_DIR = HERE / "cache"
DATA_DIR  = HERE / "data"

# Met.no requires a descriptive User-Agent — include a contact address in
# production deployments: https://api.met.no/weatherapi/termsofservice
_UA = "geobn/karmsundet (github.com/jensbremnes/geobn)"

# SSL context — use certifi bundle so macOS Python finds root certificates.
try:
    import certifi as _certifi
    _SSL_CTX: ssl.SSLContext = ssl.create_default_context(cafile=_certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


# ---------------------------------------------------------------------------
# Met.no API closure factories
# ---------------------------------------------------------------------------

def _probe_apis(lat: float, lon: float) -> None:
    """Make one test call to each Met.no endpoint and print the result.

    Exits the process if a critical endpoint is unreachable.
    """
    tests = [
        ("Oceanforecast complete",
         f"https://api.met.no/weatherapi/oceanforecast/2.0/complete?lat={lat:.4f}&lon={lon:.4f}"),
        ("Locationforecast compact",
         f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat:.4f}&lon={lon:.4f}"),
    ]
    for label, url in tests:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
                status = resp.status
                raw = resp.read()
            if not raw:
                print(f"  {label}: HTTP {status} — empty body (point may be outside model domain)")
            else:
                top_keys = list(json.loads(raw).keys())
                print(f"  {label}: HTTP {status} OK  (keys: {top_keys})")
        except urllib.error.HTTPError as exc:
            sys.exit(f"  {label}: HTTP {exc.code} {exc.reason} — aborting.")


def _make_ocean_fn(variable_name: str):
    """Return fn(lat, lon) -> float | None using the Met.no Oceanforecast API.

    Uses the ``/complete`` endpoint (Oceanforecast has no compact variant).
    Response path: ``properties.timeseries[0].data.instant.details``.
    """

    def _fn(lat: float, lon: float) -> float | None:
        url = (
            f"https://api.met.no/weatherapi/oceanforecast/2.0/complete"
            f"?lat={lat:.4f}&lon={lon:.4f}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
                data = json.loads(resp.read())
            details = (
                data["properties"]["timeseries"][0]["data"]["instant"]["details"]
            )
            return float(details[variable_name])
        except Exception:
            return None

    return _fn


def _make_loc_fn(variable_name: str, variant: str = "compact"):
    """Return fn(lat, lon) -> float | None using the Met.no Locationforecast API.

    Parameters
    ----------
    variable_name:
        Instant-details key, e.g. ``wind_speed``.
    variant:
        ``"compact"`` (default) or ``"complete"``.
        ``fog_area_fraction`` is only present in ``"complete"``.
    """
    def _fn(lat: float, lon: float) -> float | None:
        url = (
            f"https://api.met.no/weatherapi/locationforecast/2.0/{variant}"
            f"?lat={lat:.4f}&lon={lon:.4f}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
                data = json.loads(resp.read())
            details = (
                data["properties"]["timeseries"][0]["data"]["instant"]["details"]
            )
            return float(details[variable_name])
        except Exception:
            return None

    return _fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    H = round((NORTH - SOUTH) / RESOLUTION)   # 150 rows
    W = round((EAST  - WEST)  / RESOLUTION)   # 200 cols

    print("Karmsundet USV Maritime Risk — geobn demo")
    print(f"Study area  : {WEST}°E – {EAST}°E, {SOUTH}°N – {NORTH}°N")
    print(f"Grid        : {H} × {W} pixels at {RESOLUTION}° (~200 m)")

    # ── 1. Load BN and configure grid ─────────────────────────────────────
    bn = geobn.load(HERE / "usv_risk.bif")
    bn.set_grid(CRS, RESOLUTION, (WEST, SOUTH, EAST, NORTH))

    # ── 2. Fetch EMODnet bathymetry ────────────────────────────────────────
    print("\nFetching EMODnet bathymetry (cached after first run) ...")
    _EMODNET_URL = "https://ows.emodnet-bathymetry.eu/wcs"
    try:
        raw_depth = bn.fetch_raw(geobn.WCSSource(
            url=_EMODNET_URL,
            layer="emodnet:mean",
            version="2.0.1",
            valid_range=(-1000.0, 100.0),
            cache_dir=CACHE_DIR,
        ))
    except Exception as exc:
        sys.exit(f"ERROR fetching bathymetry: {exc}")

    # EMODnet convention: negative = below sea level, positive = land/above.
    # Convert to positive depth below surface; land pixels → NaN.
    depth = -raw_depth
    depth[depth < 0] = np.nan   # land pixels (originally positive in EMODnet)

    water_pixels = int(np.isfinite(depth).sum())
    print(f"Bathymetry  : {water_pixels:,} water pixels "
          f"(depth range {np.nanmin(depth):.0f}–{np.nanmax(depth):.0f} m)")

    # ── 3. Wire inputs ─────────────────────────────────────────────────────
    bn.set_input("water_depth", geobn.ArraySource(depth))

    # AIS traffic density — the pre-computed GeoTIFF where it has data, a
    # medium-traffic prior everywhere else.
    ais_path = DATA_DIR / "ais_density_karmsundet.tif"
    ais_source = geobn.MosaicSource(
        [geobn.RasterSource(ais_path), geobn.ConstantSource(2.0)],
        names=["ais_density", "medium_traffic_prior"],
        on_error="skip",
    )
    bn.set_input("vessel_traffic", ais_source)

    ais_raw, ais_provenance = bn.fetch_raw(ais_source, return_provenance=True)
    from_ais = float((ais_provenance == 0).mean())
    if from_ais:
        print(f"AIS density : {from_ais:.0%} of the grid from {ais_path.name}, "
              "the rest from the medium-traffic prior")
    else:
        print("AIS density : medium-traffic prior over the whole grid.")
        print("              Run create_ais_density.py to generate "
              f"data/{ais_path.name} from Kystverket AIS data.")

    print("\nTesting Met.no API connectivity ...")
    _centre_lat = (SOUTH + NORTH) / 2
    _centre_lon = (WEST  + EAST)  / 2
    _probe_apis(_centre_lat, _centre_lon)

    print("\nFetching Met.no forecasts (cached after the first run) ...")

    # The forecasts are sampled once and then kept: this example is about the
    # output, not about the weather being current.  Pass cache_ttl to re-sample
    # after a given age, e.g. cache_ttl=timedelta(hours=6).
    forecasts = [
        ("wave_height",   "Oceanforecast: sea_surface_wave_height",
         _make_ocean_fn("sea_surface_wave_height")),
        ("current_speed", "Oceanforecast: sea_water_speed",
         _make_ocean_fn("sea_water_speed")),
        ("wind_speed",    "Locationforecast: wind_speed",
         _make_loc_fn("wind_speed")),
        ("fog_fraction",  "Locationforecast: fog_area_fraction",
         _make_loc_fn("fog_area_fraction", variant="complete")),
    ]
    for node, description, fn in forecasts:
        print(f"  {node:<14} ({description}) ...")
        bn.set_input(node, geobn.PointGridSource(
            fn=fn,
            sample_points=5,
            delay=0.05,
            name=node,
            cache_dir=CACHE_DIR,
        ))

    # ── 4. Discretizations ────────────────────────────────────────────────
    # Compute AIS percentile thresholds from the depth array as a proxy for
    # grid shape; actual percentiles are computed from fetched traffic data
    # inside the BN pipeline at inference time — here we pre-compute from the
    # depth-valid mask to set meaningful fallback bins.
    bn.set_discretization("water_depth",     [0, 5, 20, 50, 200, 2000])
    bn.set_discretization("vessel_traffic",  [0.0, 1.0, 3.0, 1000.0])
    bn.set_discretization("wave_height",     [0.0, 0.5, 1.5, 15.0])
    bn.set_discretization("wind_speed",      [0.0, 5.0, 12.0, 50.0])
    bn.set_discretization("current_speed",   [0.0, 0.3, 1.0, 5.0])
    bn.set_discretization("fog_fraction", [0.0, 0.2, 0.6, 1.01])

    # ── 5. Precompute lookup table ─────────────────────────────────────────
    # 5 × 3 × 3 × 3 × 3 × 3 = 1,215 unique evidence combinations
    print("\nPrecomputing inference table (1,215 combos) ...")
    bn.precompute(query=["usv_risk"])

    # ── 6. Run inference ───────────────────────────────────────────────────
    print("Running BN inference ...")
    try:
        result = bn.infer(query=["usv_risk"])
    except Exception as exc:
        sys.exit(f"ERROR during inference: {exc}")

    probs = result.probabilities["usv_risk"]   # (H, W, 3)
    ent   = result.entropy("usv_risk")          # (H, W)

    # ── 7. Console statistics ──────────────────────────────────────────────
    valid_pixels = int(np.isfinite(probs[..., 0]).sum())
    if valid_pixels == 0:
        print("\nWARNING: 0 valid pixels — all inputs are NaN for every pixel.")
        print("Check that all Met.no API calls succeeded and EMODnet data covers the area.")
        sys.exit(1)

    def bar(val: float, width: int = 20) -> str:
        if not np.isfinite(val):
            return "─" * width
        filled = round(val * width)
        return "█" * filled + "░" * (width - filled)

    print(f"\n  Valid pixels : {valid_pixels:,} / {H * W:,}")
    print("\n── USV risk distribution ────────────────────────────────────────")
    for i, state in enumerate(result.state_names["usv_risk"]):
        p = float(np.nanmean(probs[..., i]))
        print(f"  P({state:8s}) mean {p:.3f}  {bar(p)}")

    print(f"\n  Mean entropy : {float(np.nanmean(ent)):.3f} bits")

    # Risk by depth zone
    p_high = probs[..., 2]
    shallow_mask  = (depth >= 0) & (depth <  5)
    moderate_mask = (depth >= 5) & (depth < 20)
    deep_mask     = depth >= 50

    print("\n── P(high usv_risk) by depth zone ───────────────────────────────")
    for label, mask in [
        ("very_shallow (0–5 m)  ", shallow_mask),
        ("shallow      (5–20 m) ", moderate_mask),
        ("deep         (≥50 m)  ", deep_mask),
    ]:
        if mask.any():
            p = float(np.nanmean(p_high[mask]))
            print(f"  {label}: {p:.3f}  {bar(p)}")

    # ── 8. Risk score (expected value) ────────────────────────────────────
    risk_score = result.expected_value("usv_risk", {"low": 10.0, "medium": 50.0, "high": 90.0})

    print(f"\nRisk score  : min={np.nanmin(risk_score):.1f}  "
          f"max={np.nanmax(risk_score):.1f}  "
          f"mean={np.nanmean(risk_score):.1f}")

    # ── 9. Export GeoTIFF ─────────────────────────────────────────────────
    result.to_geotiff(OUT_DIR, layers={"risk_score": risk_score})
    tif_path = OUT_DIR / "usv_risk.tif"
    print(f"GeoTIFF written → {tif_path}  (+ risk_score.tif)")
    print("  Band 1: P(low)  Band 2: P(medium)  Band 3: P(high)  Band 4: entropy")

    # ── 10. Interactive map ────────────────────────────────────────────────
    extra = {
        "Risk score (10–90)": risk_score,
        "Water depth (m)": depth,
    }
    # The provenance layer shows which pixels came from AIS and which from
    # the prior.
    extra["AIS traffic density"] = ais_raw
    extra["AIS source index"] = ais_provenance.astype(np.float32)

    html_path = result.show_map(
        OUT_DIR,
        filename="usv_risk_map.html",
        extra_layers=extra,
        show_probability_bands=True,
        show_category=True,
        show_entropy=True,
    )
    print(f"\nInteractive map opened in browser → {html_path}")
    print("  Use the layer control (top-right) to switch overlays.")


if __name__ == "__main__":
    main()
