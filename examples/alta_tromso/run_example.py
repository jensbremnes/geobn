"""Alta → Tromsø: a robot boat in calm weather and storm — geobn demo.

A small unmanned surface vessel (USV) sails from Alta to Tromsø in northern
Norway.  The trip goes out Altafjorden, past Stjernøya and Loppa, across
Lopphavet (about 70 km of open sea on the Troms–Finnmark border), past
Skjervøy and the mouth of Lyngen, and on to Tromsø.

geobn turns bathymetry, ship traffic and archived weather into a map of
passage risk, and a route planner then finds the path that balances distance
against that risk, much as a car's satnav routes around traffic.  The same
trip is planned for a calm summer day and for a winter storm, using the
weather that was actually observed on both days.  In the storm, the open
water of Lopphavet becomes the riskiest part of the map and the route moves
into sheltered water.

Route planning is not part of geobn; ``routing.py`` shows how a geobn risk
map is used by a planner.

Bayesian network (usv_passage.bif)
----------------------------------
    water_depth ─┬────────────────────────► grounding_risk ──┐
    wave_height ─┤ (breaking waves in shallows)               │
                 ├──────────────► sea_state_risk ─────────────┼──► usv_risk
    current_speed┘                                            │
    vessel_traffic ───────────────► collision_risk ───────────┤
    air_temperature ─┬────────────► icing_risk ───────────────┘
    wind_speed ──────┘

Data sources
------------
water_depth (DerivedSource over WCSSource)
    EMODnet Bathymetry WCS, ~115 m.  Elevation is flipped to depth; land is
    NaN, which makes it impassable for the planner.
vessel_traffic (WCSSource)
    EMODnet Human Activities vessel density, all ship types, 2023 average,
    in hours per km² per month on a ~1 km grid.
wave_height (MosaicSource)
    MET Norway WW3 4 km wave model archive where the model has sea, and a
    fetch-limited estimate (waves.py) in the sounds that it treats as land.
current_speed (MosaicSource)
    MET Norway NorKyst v3 800 m ocean model archive, surface current.
wind_speed, air_temperature (MosaicSource)
    MET Norway MEPS 2.5 km weather model archive, 10 m wind and 2 m
    temperature.

The MET Norway archives are read over OPeNDAP by ``met_archive.py``.  Each
weather mosaic ends in a constant for its scenario, used only if the
archive cannot be reached; the run reports which source supplied each
layer.

Outputs (examples/alta_tromso/output/)
--------------------------------------
    scenarios.png            calm and storm side by side, with both routes
    alta_tromso_map.html     interactive map: risk, waves and routes per scenario
    storm_timelapse.gif      the storm arriving, hour by hour, route re-planned
    <scenario>/usv_risk.tif  P(low), P(medium), P(high), entropy
    <scenario>/risk_score.tif
    routes_<scenario>.geojson

Data: EMODnet Bathymetry and EMODnet Human Activities (CC BY 4.0); MET
Norway (CC BY 4.0).

Run
---
    uv run python examples/alta_tromso/run_example.py

Add ``--no-timelapse`` to skip the GIF, which takes most of the run time.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from affine import Affine
from met_archive import MetArchiveSource
from pyproj import Transformer
from routing import plan_route, snap_to_water
from scipy.ndimage import distance_transform_edt, maximum_filter
from waves import effective_fetch, mean_direction, sheltered_hs, spread_offshore

import geobn

# This script prints arrows and box-drawing rules, which the default console
# encoding on Windows (cp1252) cannot represent.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE      = Path(__file__).parent
OUT_DIR   = HERE / "output"
CACHE_DIR = HERE / "cache"

# ---------------------------------------------------------------------------
# Study area: lon 18.4–23.6 E, lat 69.45–70.65 N, in UTM zone 34N so that
# pixel sizes and route lengths are in metres.
# ---------------------------------------------------------------------------
CRS        = "EPSG:32634"
RESOLUTION = 250.0                                     # metres
EXTENT     = (398_000, 7_704_500, 602_000, 7_840_500)  # 816 × 544 pixels

# (lon, lat) of the harbours.  Both are snapped to the nearest water pixel.
ALTA   = (23.27, 69.97)
TROMSO = (18.96, 69.65)

# ---------------------------------------------------------------------------
# Scenarios.  The times were picked by scanning the WW3 archive at a point in
# Lopphavet (70.40 N, 21.0 E).  wind_from, wind_speed, air_temperature and
# hs_offshore summarise the archive over the water of the study area at that
# time (medians, and the 90th percentile of the offshore wave height); they
# are used only if the archive cannot be reached.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    time: datetime
    note: str
    wind_from: float        # degrees, direction the wind blows from
    wind_speed: float       # m/s
    air_temperature: float  # °C
    hs_offshore: float      # m, significant wave height in open water


SCENARIOS = [
    Scenario(
        name="calm",
        label="Calm summer day",
        time=datetime(2024, 7, 20, 12, tzinfo=timezone.utc),
        # The calmest summer midday in Lopphavet in the 2024 archive: WW3
        # gives Hs 0.25 m and 1.4 m/s wind.
        note="Calmest summer midday of 2024 in the WW3 archive",
        wind_from=310.0, wind_speed=1.7, air_temperature=11.8, hs_offshore=0.4,
    ),
    Scenario(
        name="storm",
        label="Winter storm",
        time=datetime(2024, 2, 1, 16, tzinfo=timezone.utc),
        # Extreme weather "Ingunn", 31 January – 1 February 2024 (MET Norway,
        # MET info 25/2024).  At this hour WW3 gives Hs 9 m and 23 m/s wind
        # from WNW in Lopphavet, the direction the sea is most open to.
        note="Extreme weather Ingunn (MET Norway, MET info 25/2024)",
        wind_from=317.0, wind_speed=20.4, air_temperature=1.6, hs_offshore=8.2,
    ),
]

# The storm timelapse: hourly frames around the storm scenario.
TIMELAPSE_HOURS = range(-10, 7)

# ---------------------------------------------------------------------------
# Discretisation.  All state thresholds of the input nodes, in one place.
# ---------------------------------------------------------------------------
BREAKPOINTS = {
    # m: <5 very_shallow (rocks and skerries hide at this resolution),
    # 5–20 shallow (large waves break), 20–50 moderate, >50 deep
    "water_depth":     [0, 5, 20, 50, 6000],
    # Significant wave height, m: <1 calm, 1–2.5 moderate, 2.5–4.5 rough,
    # >4.5 severe for a 5–10 m vessel
    "wave_height":     [0, 1.0, 2.5, 4.5, 30],
    # Surface current, m/s: <0.3 weak, 0.3–1 moderate, >1 strong
    "current_speed":   [0, 0.3, 1.0, 10],
    # Ship hours per km² per month: <2 low, 2–20 medium (coastal routes),
    # >20 high (the busiest fairways, about the top 5 % of the water)
    "vessel_traffic":  [0, 2.0, 20.0, 1e6],
    # °C: sea spray freezes below about -2 °C
    "air_temperature": [-60, -2.0, 2.0, 45],
    # 10 m wind, m/s: <8 light, 8–15 strong, >15 gale
    "wind_speed":      [0, 8.0, 15.0, 80],
}

# ---------------------------------------------------------------------------
# Route planning.  Moving through a pixel costs its length times
#     1 + K_RISK · risk
# where risk is the expected usv_risk on a 0 (low) – 0.5 (medium) – 1 (high)
# scale.  With K_RISK = 8 a kilometre of certain high risk costs as much as
# 9 km of risk-free water, so the planner accepts a detour of up to 9 times
# the length of a high-risk stretch to avoid it.  K_RISK = 0 gives the
# shortest route.
# ---------------------------------------------------------------------------
K_RISK = 8.0
RISK_VALUES = {"low": 0.0, "medium": 0.5, "high": 1.0}

# Place names for the map, (lon, lat).
PLACES = {
    "Alta":      (23.27, 69.97),
    "Tromsø":    (18.96, 69.65),
    "Lopphavet": (21.05, 70.47),
    "Loppa":     (21.45, 70.35),
    "Skjervøy":  (20.97, 70.03),
    "Lyngen":    (20.35, 69.80),
}


# ---------------------------------------------------------------------------
# Archive files on thredds.met.no.  WW3 and MEPS run every six hours; each
# hour is read from the latest run that started at or before it.  NorKyst
# runs once a day at 00 UTC.
# ---------------------------------------------------------------------------

def _run(t: datetime) -> datetime:
    return t.replace(hour=t.hour // 6 * 6, minute=0, second=0, microsecond=0)


def ww3_file(t: datetime) -> str:
    r = _run(t)
    return f"ww3_4km_archive_files/{r:%Y/%m/%d}/ww3_4km_{r:%Y%m%dT%H}Z.nc"


def meps_file(t: datetime) -> str:
    r = _run(t)
    return f"meps25epsarchive/{r:%Y/%m/%d}/meps_det_sfc_{r:%Y%m%dT%H}Z.ncml"


def norkyst_file(t: datetime) -> str:
    return (f"fou-hi/norkystv3_his_files/{t:%Y/%m/%d}/"
            f"norkyst800_his_zdepth_{t:%Y%m%d}T00Z_m00_AN.nc")


def depth_below_surface(elevation: np.ndarray) -> np.ndarray:
    """Positive depth below the sea surface from EMODnet elevation; land → NaN."""
    return np.where(elevation <= 0, -elevation, np.nan)


def kelvin_to_celsius(t: np.ndarray) -> np.ndarray:
    return t - 273.15


# The fetch depends only on the land mask and the wind direction, and the
# DerivedSource that uses it runs on every fetch, so it is kept per direction.
_FETCH: dict[tuple[int, int], np.ndarray] = {}


def wave_height_250m(
    offshore_hs: np.ndarray,
    elevation: np.ndarray,
    wind_speed: np.ndarray,
    wind_from: np.ndarray,
) -> np.ndarray:
    """Offshore wave height scaled by each pixel's exposure (see waves.py)."""
    water = elevation <= 0
    direction = round(mean_direction(wind_from[water]) / 5) * 5 % 360
    key = (direction, hash(water.tobytes()))
    if key not in _FETCH:
        _FETCH[key] = effective_fetch(water, direction, RESOLUTION, max_fetch=100_000)
    return sheltered_hs(offshore_hs, _FETCH[key], wind_speed)


def directional_fetch(water: np.ndarray, n: int = 16) -> np.ndarray:
    """Effective fetch (m) for *n* wind directions, for the interactive map.

    Direction k is k · 360 / n degrees.  The result is cached next to the
    other downloads, keyed on the land mask, since it takes a minute or two.
    """
    key = hashlib.sha256(np.packbits(water).tobytes()).hexdigest()[:16]
    path = CACHE_DIR / f"fetch_{n}dir_{key}.npy"
    if path.exists():
        return np.load(path)
    print(f"Computing the fetch for {n} wind directions (cached after the first run) ...")
    fetch = np.stack([
        effective_fetch(water, k * 360 / n, RESOLUTION, max_fetch=100_000) for k in range(n)
    ]).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, fetch)
    return fetch


def weather_sources(sc: Scenario, t: datetime, elevation) -> dict:
    """The weather inputs at time *t*, and the mosaics they are built from.

    Every mosaic ends in a constant for the scenario, used only where the
    archive above it cannot be reached.
    """
    cache = {"cache_dir": CACHE_DIR}

    def archive(dataset, variables, name, **kw):
        return MetArchiveSource(dataset, variables, t, name=name, **cache, **kw)

    wind_speed = geobn.MosaicSource(
        [archive(meps_file(t), ("x_wind_10m", "y_wind_10m"), "wind_speed",
                 combine=np.hypot),
         geobn.ConstantSource(sc.wind_speed)],
        names=["MEPS", "scenario constant"], on_error="skip",
    )
    wind_from = geobn.MosaicSource(
        [archive(meps_file(t), "wind_direction", "wind_direction"),
         geobn.ConstantSource(sc.wind_from)],
        names=["MEPS", "scenario constant"], on_error="skip",
    )
    air_temperature = geobn.MosaicSource(
        [geobn.DerivedSource(kelvin_to_celsius,
                             archive(meps_file(t), "air_temperature_2m", "air_temperature")),
         geobn.ConstantSource(sc.air_temperature)],
        names=["MEPS", "scenario constant"], on_error="skip",
    )
    # The offshore sea state from WW3, carried into the sounds the 4 km model
    # does not resolve, then scaled by how exposed each 250 m pixel is.
    offshore_hs = geobn.MosaicSource(
        [geobn.DerivedSource(spread_offshore, archive(ww3_file(t), "hs", "wave_height")),
         geobn.ConstantSource(sc.hs_offshore)],
        names=["WW3 4 km", "scenario constant"], on_error="skip",
    )
    wave_height = geobn.DerivedSource(
        wave_height_250m, offshore_hs, elevation, wind_speed, wind_from
    )
    # NorKyst, bilinear first.  Bilinear resampling leaves a gap along the
    # coast where a model cell has a land neighbour; the nearest model cell
    # fills it.  Both read the same cache entry.
    norkyst = dict(dataset=norkyst_file(t), variables=("u_eastward", "v_northward"),
                   name="current_speed", combine=np.hypot)
    current_speed = geobn.MosaicSource(
        [archive(**norkyst),
         archive(**norkyst, resampling="nearest"),
         geobn.ConstantSource(0.2)],
        names=["NorKyst 800 m", "NorKyst 800 m (nearest cell)", "assumed 0.2 m/s"],
        on_error="skip",
    )
    inputs = {
        "wave_height": wave_height,
        "current_speed": current_speed,
        "wind_speed": wind_speed,
        "air_temperature": air_temperature,
    }
    # The mosaics the inputs are built from, for the provenance report.
    origins = {
        "offshore waves": offshore_hs,
        "current_speed": current_speed,
        "wind_speed": wind_speed,
        "wind direction": wind_from,
        "air_temperature": air_temperature,
    }
    return inputs, origins


def report_provenance(bn, sources: dict, water: np.ndarray) -> dict:
    """Print the share of water pixels each source supplied, per input."""
    shares = {}
    for node, src in sources.items():
        _, prov = bn.fetch_raw(src, return_provenance=True)
        parts = []
        for i, name in enumerate(src.names):
            share = float((prov[water] == i).mean())
            if share > 0:
                parts.append(f"{name} {share:.0%}")
        shares[node] = parts
        print(f"    {node:<16}: " + ", ".join(parts))
    return shares


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@dataclass
class Route:
    kind: str
    path: np.ndarray         # (N, 2) row, col
    length_km: float
    mean_risk: float
    max_p_high: float


def route_stats(kind, path, length_m, risk, p_high) -> Route:
    r, c = path[:, 0], path[:, 1]
    # Weight each pixel by the length of the steps next to it.
    seg = np.hypot(*np.diff(path, axis=0).T)
    w = np.zeros(len(path))
    w[:-1] += seg / 2
    w[1:] += seg / 2
    return Route(
        kind=kind, path=path, length_km=length_m / 1000,
        mean_risk=float(np.sum(w * risk[r, c]) / np.sum(w)),
        max_p_high=float(np.max(p_high[r, c])),
    )


def plan_routes(risk, p_high, passable, start, goal) -> dict[str, Route]:
    base = np.where(passable, 1.0, np.nan)
    costs = {
        "shortest": base,
        "risk_aware": base + K_RISK * risk,
    }
    routes = {}
    for kind, cost in costs.items():
        path, length_m, _ = plan_route(cost, start, goal, RESOLUTION)
        routes[kind] = route_stats(kind, path, length_m, risk, p_high)
    return routes


def pixel_to_lonlat(bn_transform, path: np.ndarray) -> list[list[float]]:
    xs, ys = bn_transform * (path[:, 1] + 0.5, path[:, 0] + 0.5)
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(xs, ys)
    return [[round(a, 5), round(b, 5)] for a, b in zip(lon, lat)]


def write_geojson(path: Path, sc: Scenario, routes: dict[str, Route], transform) -> None:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": pixel_to_lonlat(transform, r.path)},
            "properties": {
                "scenario": sc.name, "time": sc.time.isoformat(), "kind": r.kind,
                "length_km": round(r.length_km, 1),
                "mean_risk": round(r.mean_risk, 3),
                "max_p_high": round(r.max_p_high, 3),
            },
        }
        for r in routes.values()
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=1))


def narrowest_point(
    path: np.ndarray, passable: np.ndarray, skip: int = 8
) -> tuple[float, tuple[int, int]]:
    """Approximate channel width (pixels) at the narrowest point of *path*.

    The width at a pixel is twice the largest distance to land within 4
    pixels of it, so a route that hugs the shore of a wide fjord still counts
    the fjord's width; widths of 9 pixels or more are all reported as 9.  The
    first and last *skip* pixels are left out: the harbours are narrow by
    nature.
    """
    width = 2 * maximum_filter(distance_transform_edt(passable), size=9) - 1
    path = path[skip:-skip]
    w = width[path[:, 0], path[:, 1]]
    i = int(np.argmin(w))
    return float(w[i]), (int(path[i, 0]), int(path[i, 1]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # The mosaics warn when a source cannot be reached; the provenance report
    # below says the same thing per layer, so one line each is enough.
    warnings.simplefilter("once", UserWarning)

    geobn.set_verbose(False)
    print("Alta → Tromsø: a robot boat in calm weather and storm — geobn demo")

    # ── 1. Network and grid ────────────────────────────────────────────────
    bn = geobn.load(HERE / "usv_passage.bif")
    bn.set_grid(CRS, RESOLUTION, EXTENT)
    H, W = round((EXTENT[3] - EXTENT[1]) / RESOLUTION), round((EXTENT[2] - EXTENT[0]) / RESOLUTION)
    # The pixel-to-UTM transform of that grid, for placing points and routes.
    transform = Affine(RESOLUTION, 0, EXTENT[0], 0, -RESOLUTION, EXTENT[3])
    print(f"Grid        : {H} × {W} pixels at {RESOLUTION:.0f} m ({CRS})")

    # ── 2. Static inputs: bathymetry and ship traffic ──────────────────────
    elevation = geobn.WCSSource(
        url="https://ows.emodnet-bathymetry.eu/wcs",
        layer="emodnet:mean",
        version="2.0.1",
        valid_range=(-1000.0, 3000.0),
        cache_dir=CACHE_DIR,
    )
    water_depth = geobn.DerivedSource(depth_below_surface, elevation)
    vessel_traffic = geobn.WCSSource(
        url="https://ows.emodnet-humanactivities.eu/wcs",
        layer="emodnet__vesseldensity_allavg",
        extra_subsets=['time("2023-01-01T00:00:00.000Z")'],
        cache_dir=CACHE_DIR,
    )

    print("\nFetching EMODnet bathymetry and vessel density (cached after the first run) ...")
    elev = bn.fetch_raw(elevation)
    depth = depth_below_surface(elev)
    water = np.isfinite(depth)
    print(f"Bathymetry  : {int(water.sum()):,} water pixels, depth up to {np.nanmax(depth):.0f} m")
    traffic = bn.fetch_raw(vessel_traffic)
    print(f"Traffic     : median {np.nanmedian(traffic[water]):.2f}, "
          f"95th percentile {np.nanpercentile(traffic[water], 95):.1f} h/km²/month")

    bn.set_input("water_depth", water_depth)
    # Where EMODnet has no density value on water, there is no recorded traffic.
    bn.set_input("vessel_traffic", geobn.MosaicSource(
        [vessel_traffic, geobn.ConstantSource(0.0)], names=["EMODnet", "none recorded"],
    ))

    # ── 3. Start and goal ──────────────────────────────────────────────────
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

    def lonlat_to_rc(lon, lat):
        x, y = to_utm.transform(lon, lat)
        col, row = ~transform * (x, y)
        return int(row), int(col)

    ends = {}
    for name, lonlat in (("Alta", ALTA), ("Tromsø", TROMSO)):
        rc = lonlat_to_rc(*lonlat)
        snapped = snap_to_water(water, rc)
        moved = np.hypot(rc[0] - snapped[0], rc[1] - snapped[1]) * RESOLUTION
        ends[name] = snapped
        print(f"{name:<12}: pixel {snapped}" + (f" (moved {moved:.0f} m to water)" if moved else ""))

    # ── 4. Discretisation, frozen static inputs, lookup table ──────────────
    for node, bp in BREAKPOINTS.items():
        bn.set_discretization(node, bp)
    for node, src in weather_sources(SCENARIOS[0], SCENARIOS[0].time, elevation)[0].items():
        bn.set_input(node, src)
    bn.freeze("water_depth", "vessel_traffic")
    print("\nPrecomputing the inference table (4·4·3·3·3·3 = 1,296 combinations) ...")
    bn.precompute(query=["usv_risk"])

    # ── 5. Scenarios ───────────────────────────────────────────────────────
    results = {}
    for sc in SCENARIOS:
        print(f"\n── {sc.label}: {sc.time:%d %b %Y %H:%M} UTC ──────────────────────")
        print(f"    {sc.note}")
        inputs, origins = weather_sources(sc, sc.time, elevation)
        for node, src in inputs.items():
            bn.set_input(node, src)
        provenance = report_provenance(bn, origins, water)
        wave = bn.fetch_raw(inputs["wave_height"])
        print(f"    wave height     : max {np.nanmax(wave[water]):.1f} m, "
              f"median {np.nanmedian(wave[water]):.1f} m")

        result = bn.infer(query=["usv_risk"])
        risk = result.expected_value("usv_risk", RISK_VALUES)
        p_high = result.probabilities["usv_risk"][..., 2]
        passable = water & np.isfinite(risk)

        routes = plan_routes(risk, p_high, passable, ends["Alta"], ends["Tromsø"])
        width, where = narrowest_point(routes["risk_aware"].path, passable)
        lon, lat = pixel_to_lonlat(transform, np.array([where]))[0]
        print(f"    narrowest water on the risk-aware route: about {width:.0f} pixels "
              f"({width * RESOLUTION / 1000:.1f} km) at {lat:.3f} N, {lon:.3f} E")

        result.to_geotiff(OUT_DIR / sc.name, layers={"risk_score": risk})
        write_geojson(OUT_DIR / f"routes_{sc.name}.geojson", sc, routes, transform)
        results[sc.name] = dict(scenario=sc, risk=risk, routes=routes, wave=wave,
                                current=bn.fetch_raw(inputs["current_speed"]),
                                provenance=provenance)

    # ── 6. Summary ─────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  {'scenario':<8} {'route':<19} {'length':>9} {'mean risk':>10} {'max P(high)':>12}")
    for name, res in results.items():
        for r in res["routes"].values():
            print(f"  {name:<8} {r.kind:<19} {r.length_km:>6.0f} km {r.mean_risk:>10.2f} "
                  f"{r.max_p_high:>12.2f}")

    # ── 7. Figures ─────────────────────────────────────────────────────────
    from figures import hero_figure, timelapse  # noqa: PLC0415
    from webmap import build_payload, write_map  # noqa: PLC0415

    places = {k: lonlat_to_rc(*v) for k, v in PLACES.items()}
    hero = hero_figure(OUT_DIR / "scenarios.png", elev, results, ends, places)
    print(f"\nFigure      → {hero}")

    # The interactive map recomputes risk and route in the browser for any
    # weather set with its sliders; see webmap.py for what it carries.
    payload = build_payload(
        bn=bn,
        breakpoints=BREAKPOINTS,
        risk_weights=list(RISK_VALUES.values()),
        k_risk=K_RISK,
        depth=depth,
        traffic=np.where(np.isfinite(traffic), traffic, 0.0),   # as the mosaic fills it
        current=results[SCENARIOS[0].name]["current"],
        fetch=directional_fetch(elev <= 0),
        shortest=results[SCENARIOS[0].name]["routes"]["shortest"].path,
        ends=ends,
        # The sliders start at the storm's weather.
        initial=dict(hs=SCENARIOS[-1].hs_offshore, u=SCENARIOS[-1].wind_speed,
                     dir=SCENARIOS[-1].wind_from, t=SCENARIOS[-1].air_temperature),
        crs=CRS,
        transform=transform,
        pixel_size=RESOLUTION,
    )
    web = write_map(OUT_DIR / "alta_tromso_map.html", payload)
    print(f"Web map     → {web}  ({web.stat().st_size / 1e6:.1f} MB)")

    if "--no-timelapse" in sys.argv:
        return
    storm = SCENARIOS[-1]
    frames = []
    print(f"\nStorm timelapse ({len(TIMELAPSE_HOURS)} frames) ...")
    for dh in TIMELAPSE_HOURS:
        t = storm.time + timedelta(hours=dh)
        for node, src in weather_sources(storm, t, elevation)[0].items():
            bn.set_input(node, src)
        result = bn.infer(query=["usv_risk"])
        risk = result.expected_value("usv_risk", RISK_VALUES)
        p_high = result.probabilities["usv_risk"][..., 2]
        routes = plan_routes(risk, p_high, water & np.isfinite(risk),
                             ends["Alta"], ends["Tromsø"])
        frames.append((t, risk, routes))
        print(f"  {t:%d %b %H:%M} UTC: risk-aware route {routes['risk_aware'].length_km:.0f} km")
    gif = timelapse(OUT_DIR / "storm_timelapse.gif", elev, frames, ends, places)
    print(f"Timelapse   → {gif}")


if __name__ == "__main__":
    main()
