"""Interactive Leaflet map for the Alta → Tromsø example.

The map has sliders for the weather: offshore wave height, wind speed, wind
direction and air temperature.  Moving one recomputes the passage risk for
every pixel and plans the risk-aware route again, in the browser.

The browser runs the same model as the Python script, without Python or
pgmpy:

- geobn's precomputed table, exported here as the expected ``usv_risk`` and
  P(high) for each of the 1,296 combinations of input states;
- the input states that do not change with the sliders: water depth, ship
  traffic and surface current (the calm-day NorKyst field);
- the effective fetch of every pixel for 16 wind directions (one byte per
  pixel, see ``quantise_fetch``), from which
  ``waves.sheltered_hs`` gives the wave height for any offshore wave height
  and wind speed;
- the discretisation breakpoints and the route cost ``1 + K_RISK · risk``.

The wind and temperature from the sliders apply to the whole area, and the
offshore wave height is the same everywhere before sheltering; this is the
same calculation the script falls back to when the weather archives cannot
be reached.  The script (``webmap.js``) looks up the risk, draws it, and
runs Dijkstra's algorithm on the same 8-connected graph as ``routing.py``.

Leaflet draws image overlays in Web Mercator, so the page carries two
coarse lattices, computed here with pyproj: Web Mercator pixel → grid pixel
for drawing the risk, and grid pixel → longitude and latitude for drawing
the routes.
"""
from __future__ import annotations

import base64
import itertools
import json
import zlib
from pathlib import Path

import folium
import numpy as np
from affine import Affine
from figures import RISK_CMAP
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer
from rasterio.warp import calculate_default_transform

HERE = Path(__file__).parent

# Order of the input nodes in the exported table; the page indexes it the
# same way.
TABLE_NODES = [
    "water_depth", "wave_height", "current_speed",
    "vessel_traffic", "air_temperature", "wind_speed",
]

WAVE_CMAP = LinearSegmentedColormap.from_list(
    "waves", ["#e8f1fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
)
WAVE_MAX = 8.0          # m, top of the wave colour scale
LATTICE_STEP = 16       # pixels between lattice points
# The fetch is stored as round(255 · sqrt(F / FETCH_MAX)) in one byte: both the
# exposure and the fetch-limited wave height depend on sqrt(F).
FETCH_MAX = 100_000.0   # m, the longest fetch (waves.effective_fetch max_fetch)

# The greyscale map comes first: the risk colours read best on it.
BASEMAPS = [
    dict(name="Kartverket greyscale",
         tiles="https://cache.kartverket.no/v1/wmts/1.0.0/topograatone/default/webmercator/{z}/{y}/{x}.png",
         attr='© <a href="https://www.kartverket.no/">Kartverket</a>'),
    dict(name="Kartverket topographic",
         tiles="https://cache.kartverket.no/v1/wmts/1.0.0/topo/default/webmercator/{z}/{y}/{x}.png",
         attr='© <a href="https://www.kartverket.no/">Kartverket</a>'),
    dict(name="Kartverket nautical chart",
         tiles="https://cache.kartverket.no/v1/wmts/1.0.0/sjokartraster/default/webmercator/{z}/{y}/{x}.png",
         attr='© <a href="https://www.kartverket.no/">Kartverket</a>'),
    dict(name="OpenStreetMap", tiles="OpenStreetMap", attr=None),
]


# ---------------------------------------------------------------------------
# Data for the page
# ---------------------------------------------------------------------------

def risk_table(bn, breakpoints: dict, weights: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Expected ``usv_risk`` and P(high) for every combination of input states.

    The rows follow ``itertools.product`` over ``TABLE_NODES``, the last node
    varying fastest.  Each state is represented by the middle of its bin, so
    ``bn.query_batch`` puts it back in that state.
    """
    sizes = [len(breakpoints[n]) - 1 for n in TABLE_NODES]
    mids = {n: [(bp[i] + bp[i + 1]) / 2 for i in range(len(bp) - 1)]
            for n, bp in breakpoints.items()}
    combos = np.array(list(itertools.product(*(range(s) for s in sizes))))
    evidence = {n: [mids[n][s] for s in combos[:, k]] for k, n in enumerate(TABLE_NODES)}
    probs = bn.query_batch(evidence, query=["usv_risk"])["usv_risk"]
    return probs @ np.asarray(weights), probs[:, -1]


def states(values: np.ndarray, breakpoints: list[float]) -> np.ndarray:
    """State index per pixel, as ``geobn`` discretises (values outside clip)."""
    return np.digitize(values, breakpoints[1:-1]).astype(np.uint8)


def quantise_fetch(fetch: np.ndarray) -> np.ndarray:
    """Fetch (m) as one byte per pixel: round(255 · sqrt(F / FETCH_MAX))."""
    q = np.round(255 * np.sqrt(np.clip(np.nan_to_num(fetch), 0, FETCH_MAX) / FETCH_MAX))
    return q.astype(np.uint8)


def dequantise_fetch(q: np.ndarray) -> np.ndarray:
    """The fetch (m) the page computes with, from :func:`quantise_fetch`."""
    return (q.astype(np.float64) / 255) ** 2 * FETCH_MAX


def _packed(array: np.ndarray) -> str:
    """zlib-compressed, base64-encoded bytes; the page inflates them."""
    return base64.b64encode(zlib.compress(np.ascontiguousarray(array).tobytes(), 9)).decode()


def _raw(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def _lattice_positions(n: int) -> np.ndarray:
    pos = np.arange(0, n, LATTICE_STEP)
    return np.unique(np.append(pos, n - 1))


def _colour_lut(cmap, alpha) -> list[int]:
    v = np.linspace(0, 1, 256)
    rgba = cmap(v)
    rgba[:, 3] = alpha(v)
    return np.round(rgba * 255).astype(np.uint8).ravel().tolist()


def _web_mercator_grid(crs: str, transform: Affine, shape: tuple[int, int]) -> dict:
    """Web Mercator overlay grid, with a lattice back to grid pixels."""
    h, w = shape
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    dst, dst_w, dst_h = calculate_default_transform(
        crs, "EPSG:3857", w, h, left=left, bottom=bottom, right=right, top=top
    )
    rows, cols = _lattice_positions(dst_h), _lattice_positions(dst_w)
    cc, rr = np.meshgrid(cols + 0.5, rows + 0.5)
    mx, my = dst * (cc, rr)
    ux, uy = Transformer.from_crs("EPSG:3857", crs, always_xy=True).transform(mx, my)
    src_col, src_row = ~transform * (ux, uy)

    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, north = to_lonlat.transform(*(dst * (0, 0)))
    east, south = to_lonlat.transform(*(dst * (dst_w, dst_h)))
    return {
        "shape": [dst_h, dst_w],
        "rows": rows.tolist(), "cols": cols.tolist(),
        "src_row": _raw(src_row.astype("<f4")), "src_col": _raw(src_col.astype("<f4")),
        "bounds": [[south, west], [north, east]],
    }


def _lonlat_lattice(crs: str, transform: Affine, shape: tuple[int, int]) -> dict:
    """Longitude and latitude of grid pixel centres on a coarse lattice."""
    h, w = shape
    rows, cols = _lattice_positions(h), _lattice_positions(w)
    cc, rr = np.meshgrid(cols + 0.5, rows + 0.5)
    x, y = transform * (cc, rr)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    return {"rows": rows.tolist(), "cols": cols.tolist(),
            "lat": _raw(lat.astype("<f8")), "lon": _raw(lon.astype("<f8"))}


def build_payload(
    *,
    bn,
    breakpoints: dict,
    risk_weights: list[float],
    k_risk: float,
    depth: np.ndarray,
    traffic: np.ndarray,
    current: np.ndarray,
    fetch: np.ndarray,
    shortest: np.ndarray,
    ends: dict,
    initial: dict,
    crs: str,
    transform: Affine,
    pixel_size: float,
) -> dict:
    """Everything the page needs, as JSON-serialisable values."""
    h, w = depth.shape
    valid = np.isfinite(depth) & np.isfinite(traffic) & np.isfinite(current)
    depth_state = states(np.nan_to_num(depth), breakpoints["water_depth"])
    depth_state[~valid] = 255   # not water, or no data: no route crosses it
    risk, p_high = risk_table(bn, breakpoints, risk_weights)
    fetch_bytes = quantise_fetch(fetch)

    return {
        "shape": [h, w],
        "pixelSize": pixel_size,
        "kRisk": k_risk,
        "nodes": TABLE_NODES,
        "sizes": [len(breakpoints[n]) - 1 for n in TABLE_NODES],
        "breakpoints": {n: list(map(float, bp)) for n, bp in breakpoints.items()},
        "tableRisk": _raw(risk.astype("<f4")),
        "tableHigh": _raw(p_high.astype("<f4")),
        "depthState": _packed(depth_state),
        "trafficState": _packed(states(np.nan_to_num(traffic), breakpoints["vessel_traffic"])),
        "currentState": _packed(states(np.nan_to_num(current), breakpoints["current_speed"])),
        "fetchMax": FETCH_MAX,
        "fetch": [_packed(f) for f in fetch_bytes],
        "shortest": _raw((shortest[:, 0] * w + shortest[:, 1]).astype("<i4")),
        "start": int(ends["Alta"][0] * w + ends["Alta"][1]),
        "goal": int(ends["Tromsø"][0] * w + ends["Tromsø"][1]),
        "initial": initial,
        # Low risk stays see-through; opacity grows up to high risk.
        "riskLut": _colour_lut(RISK_CMAP, lambda v: np.clip((v - 0.08) / 0.6, 0, 1) ** 0.7 * 0.8),
        "waveLut": _colour_lut(WAVE_CMAP, lambda v: np.full_like(v, 0.8)),
        "waveMax": WAVE_MAX,
        "overlay": _web_mercator_grid(crs, transform, (h, w)),
        "lonlat": _lonlat_lattice(crs, transform, (h, w)),
    }


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

PANEL = """
<div id="alta-panel">
  <div class="alta-title">A USV from Alta to Tromsø</div>
  <label>Offshore wave height <output id="out-hs"></output>
    <input type="range" id="in-hs" min="0" max="12" step="0.1"></label>
  <label>Wind speed <output id="out-u"></output>
    <input type="range" id="in-u" min="0" max="35" step="0.5"></label>
  <label>Wind from <output id="out-dir"></output>
    <input type="range" id="in-dir" min="0" max="337.5" step="22.5"></label>
  <label>Air temperature <output id="out-t"></output>
    <input type="range" id="in-t" min="-20" max="20" step="0.5"></label>
  <div class="alta-colour">Colour the sea by
    <label><input type="radio" name="colour" value="risk" checked> passage risk</label>
    <label><input type="radio" name="colour" value="waves"> wave height</label>
  </div>
  <div id="alta-stats">Loading…</div>
  <div class="alta-legend">
    <div id="legend-ramp"></div>
    <div id="legend-ticks"></div>
    <div class="alta-line"><svg width="34" height="8"><line x1="1" y1="4" x2="33" y2="4"
      stroke="#0b0b0b" stroke-width="3.5" stroke-linecap="round"/></svg>Risk-aware route</div>
    <div class="alta-line"><svg width="34" height="8"><line x1="1" y1="4" x2="33" y2="4"
      stroke="#52514e" stroke-width="2.5" stroke-dasharray="8 6"/></svg>Shortest route</div>
  </div>
</div>
<pre id="selftest" hidden></pre>
"""

STYLE = """
<style>
#alta-panel { position: fixed; top: 12px; left: 56px; z-index: 1000; width: 300px;
  max-height: calc(100vh - 40px); overflow-y: auto; background: rgba(255,255,255,0.95);
  padding: 12px 14px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.25);
  font: 13px/1.4 system-ui, sans-serif; color: #0b0b0b; }
#alta-panel .alta-title { font-weight: 600; font-size: 15px; margin-bottom: 4px; }
#alta-panel label { display: block; margin: 6px 0; }
#alta-panel output { float: right; color: #52514e; font-variant-numeric: tabular-nums; }
#alta-panel input[type=range] { width: 100%; margin: 2px 0 0; }
#alta-panel .alta-colour { margin: 8px 0 4px; color: #52514e; }
#alta-panel .alta-colour label { display: inline; margin: 0 6px 0 4px; color: #0b0b0b; }
#alta-stats { margin: 8px 0; padding: 8px 10px; background: #f6f5f2; border-radius: 4px; }
#legend-ramp { width: 100%; height: 10px; border-radius: 3px; }
#legend-ticks { display: flex; justify-content: space-between; color: #52514e; font-size: 12px;
  margin-bottom: 6px; }
#alta-panel .alta-line { display: flex; align-items: center; gap: 8px; margin: 2px 0; }
</style>
"""


def write_map(path: Path, payload: dict) -> Path:
    """Write the interactive map to *path* and return it."""
    grid = payload["lonlat"]
    lat = np.frombuffer(base64.b64decode(grid["lat"]), "<f8")
    lon = np.frombuffer(base64.b64decode(grid["lon"]), "<f8")
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=8,
                   tiles=None, control_scale=True)
    for i, base in enumerate(BASEMAPS):
        folium.TileLayer(tiles=base["tiles"], attr=base["attr"], name=base["name"],
                         max_zoom=18, show=i == 0).add_to(m)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    root = m.get_root()
    root.header.add_child(folium.Element(STYLE))
    root.html.add_child(folium.Element(PANEL))
    data = dict(payload, mapName=m.get_name())
    root.html.add_child(folium.Element(
        '<script type="application/json" id="alta-data">'
        + json.dumps(data).replace("</", "<\\/") + "</script>"
    ))
    # The page script runs after folium has created the map.
    root.script.add_child(folium.Element((HERE / "webmap.js").read_text(encoding="utf-8")))
    m.save(str(path))
    return path
