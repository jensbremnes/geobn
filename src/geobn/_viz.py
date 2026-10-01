"""Interactive Leaflet map visualisation for InferenceResult."""
from __future__ import annotations

import base64
import math
import webbrowser
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from affine import Affine

if TYPE_CHECKING:
    from .result import InferenceResult

# Latitude limit of Web Mercator (EPSG:3857), in degrees.
_WEB_MERCATOR_MAX_LAT = 85.0511287798

# Discrete category colours — shared by _discrete_array_to_png_url and colorbar builder.
_DISCRETE_PALETTE_HEX = ["#2ecc71", "#e6801f", "#e84c3d", "#8f479a", "#3396dc"]


def _web_mercator_grid(
    crs: str, transform: Affine, shape: tuple[int, int]
) -> tuple[Affine, tuple[int, int], list[list[float]]]:
    """Return the Web Mercator grid a result grid is displayed on.

    Leaflet places an image overlay linearly in Web Mercator, so every layer
    is resampled onto an axis-aligned EPSG:3857 grid covering the result
    grid, at about the same resolution.  Returns ``(transform, shape,
    bounds)`` where *bounds* is ``[[south, west], [north, east]]`` in degrees.
    """
    from pyproj import CRS, Transformer
    from rasterio.warp import calculate_default_transform

    from .grid import GridSpec

    H, W = shape
    lon_min, lat_min, lon_max, lat_max = GridSpec(crs, transform, shape).extent_wgs84()
    if max(abs(lat_min), abs(lat_max)) > _WEB_MERCATOR_MAX_LAT:
        raise ValueError(
            f"The result grid spans latitudes {lat_min:.4f} to {lat_max:.4f}. "
            "show_map() draws on a Web Mercator base map, which covers latitudes "
            f"up to ±{_WEB_MERCATOR_MAX_LAT:.4f}° only."
        )

    if (
        CRS.from_user_input(crs) == CRS.from_epsg(3857)
        and transform.b == 0 and transform.d == 0 and transform.e < 0
    ):
        dst, dst_shape = transform, (H, W)
    else:
        # Resolution as GDAL picks it; extent from points along every edge,
        # rounded up to whole pixels so no part of the grid is cut off.
        xs, ys = transform * (np.array([0, W, W, 0]), np.array([0, 0, H, H]))
        default, _, _ = calculate_default_transform(
            crs, "EPSG:3857", W, H,
            left=float(xs.min()), bottom=float(ys.min()),
            right=float(xs.max()), top=float(ys.max()),
        )
        res = default.a
        t = np.linspace(0.0, 1.0, 21)
        cols = np.concatenate([t * W, np.full_like(t, W), t * W, np.zeros_like(t)])
        rows = np.concatenate([np.zeros_like(t), t * H, np.full_like(t, H), t * H])
        mx, my = Transformer.from_crs(crs, "EPSG:3857", always_xy=True).transform(
            *(transform * (cols, rows))
        )
        left, top = float(np.min(mx)), float(np.max(my))
        dst_w = max(1, math.ceil((float(np.max(mx)) - left) / res))
        dst_h = max(1, math.ceil((top - float(np.min(my))) / res))
        dst = Affine(res, 0, left, 0, -res, top)
        dst_shape = (dst_h, dst_w)

    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, north = to_lonlat.transform(*(dst * (0, 0)))
    east, south = to_lonlat.transform(*(dst * (dst_shape[1], dst_shape[0])))
    return dst, dst_shape, [[south, west], [north, east]]


def _web_mercator_warper(
    crs: str, transform: Affine, dst_transform: Affine, dst_shape: tuple[int, int]
):
    """Return a function that resamples (H, W) arrays onto the Web Mercator grid.

    Nearest neighbour: each display pixel shows the value of one grid cell.
    Display pixels outside the result grid are NaN.
    """
    from pyproj import CRS

    from .grid import _nearest_resample, _PixelMapper

    if transform == dst_transform and CRS.from_user_input(crs) == CRS.from_epsg(3857):
        return lambda arr: np.asarray(arr, dtype=np.float32)
    H, W = dst_shape
    col, row = np.meshgrid(
        np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5
    )
    src_row, src_col = _PixelMapper(crs, transform, "EPSG:3857", dst_transform)(row, col)

    def warp(arr: np.ndarray) -> np.ndarray:
        return _nearest_resample(np.asarray(arr, dtype=np.float32), src_row, src_col)

    return warp


def _hex_to_rgb_float(hex_color: str) -> tuple[float, float, float]:
    """Convert a hex color string like ``#rrggbb`` to float (r, g, b) in [0, 1]."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255


def _array_to_png_url(
    arr: np.ndarray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    alpha: float = 0.65,
) -> str:
    """Return a base64 PNG data URL for use as a folium ImageOverlay image.

    NaN pixels get alpha=0 (transparent); valid pixels get *alpha*.
    Uses ``plt.imsave`` — matplotlib only, no Pillow needed.
    """
    import matplotlib.pyplot as plt

    safe_range = vmax - vmin if (not np.isnan(vmin) and vmax != vmin) else 1.0
    norm = np.clip((arr - vmin) / safe_range, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm).astype(np.float64)  # (H, W, 4)

    nan_mask = np.isnan(arr)
    rgba[nan_mask, 3] = 0.0
    rgba[~nan_mask, 3] = alpha

    buf = BytesIO()
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _discrete_array_to_png_url(
    category: np.ndarray,
    n_states: int,
    alpha: float = 0.65,
) -> str:
    """Return a base64 PNG for a discrete category array (integer 0…n_states-1)."""
    import matplotlib.pyplot as plt

    rgba = np.zeros((*category.shape, 4), dtype=np.float64)
    for i in range(min(n_states, len(_DISCRETE_PALETTE_HEX))):
        r, g, b = _hex_to_rgb_float(_DISCRETE_PALETTE_HEX[i])
        rgba[category == i] = (r, g, b, alpha)
    # NaN pixels stay at alpha=0 (zeros initialisation)

    buf = BytesIO()
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _cmap_to_hex(cmap_name: str, n: int = 6) -> list[str]:
    """Return *n* evenly-spaced hex colours from a matplotlib colormap."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    steps = max(n, 2)
    return [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _ in (cmap(i / (steps - 1)) for i in range(steps))
    ][:n]


def show_map(
    result: "InferenceResult",
    output_dir: str | Path = ".",
    filename: str = "map.html",
    overlay_opacity: float = 0.65,
    open_browser: bool = True,
    extra_layers: dict[str, np.ndarray] | None = None,
    show_probability_bands: bool = True,
    show_category: bool = True,
    show_entropy: bool = True,
) -> Path:
    """Generate and optionally open an interactive Leaflet map.

    Parameters
    ----------
    result:
        :class:`~geobn.InferenceResult` from
        :meth:`~geobn.GeoBayesianNetwork.infer`.
    output_dir:
        Directory to write the HTML file into.
    filename:
        Output filename (default ``map.html``).
    overlay_opacity:
        Opacity of probability overlays (0–1).
    open_browser:
        If True (default), open the map in the default browser.
    extra_layers:
        Additional named (H, W) arrays to include as overlays
        (e.g. ``{"Slope angle (°)": slope_deg}``).

    Each layer is resampled (nearest neighbour) onto a Web Mercator grid of
    about the same resolution before it is drawn, so that it lines up with the
    base map for any grid CRS.  The grid must lie within the latitude range
    of Web Mercator (±85.05°); otherwise ``ValueError`` is raised.

    Returns
    -------
    Path
        Path to the written HTML file.
    """
    import folium

    # ── Web Mercator display grid ─────────────────────────────────────────
    probs_any = next(iter(result.probabilities.values()))
    shape = probs_any.shape[:2]
    dst_transform, dst_shape, bounds = _web_mercator_grid(result.crs, result.transform, shape)
    warp = _web_mercator_warper(result.crs, result.transform, dst_transform, dst_shape)
    (south, west), (north, east) = bounds
    center = [(south + north) / 2, (west + east) / 2]

    # ── Map + basemaps ─────────────────────────────────────────────────────
    m = folium.Map(location=center, zoom_start=9, control_scale=True, tiles=None)

    folium.TileLayer(
        tiles="https://tile.opentopomap.org/{z}/{x}/{y}.png",
        attr=(
            'Map data: © <a href="https://www.openstreetmap.org/copyright">'
            "OpenStreetMap</a> contributors, "
            '<a href="http://viewfinderpanoramas.org">SRTM</a> | '
            'Map style: © <a href="https://opentopomap.org">OpenTopoMap</a>'
        ),
        name="OpenTopoMap",
        max_zoom=17,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://cache.kartverket.no/v1/wmts/1.0.0/topo/default/webmercator/{z}/{y}/{x}.png",
        attr='© <a href="https://www.kartverket.no/">Kartverket</a>',
        name="Kartverket Topo (Norway)",
        max_zoom=18,
        show=False,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)

    # ── Inference result layers ────────────────────────────────────────────
    import json

    layer_colorbars: dict[str, dict] = {}

    for node, probs in result.probabilities.items():
        n_states = probs.shape[-1]
        states = result.state_names[node]

        # ── Individual probability bands (all hidden) ───────────────────
        if show_probability_bands:
            state_cmaps = {0: "YlGn", n_states - 1: "YlOrRd"}   # first=green, last=red
            for i, state in enumerate(states):
                cmap_name = state_cmaps.get(i, "YlOrBr")
                layer_name = f"P({state})"
                img_url = _array_to_png_url(warp(probs[..., i]), cmap_name, 0.0, 1.0, overlay_opacity)
                fg = folium.FeatureGroup(name=layer_name, show=False)
                folium.raster_layers.ImageOverlay(image=img_url, bounds=bounds, opacity=1.0).add_to(fg)
                fg.add_to(m)
                layer_colorbars[layer_name] = {
                    "title": layer_name,
                    "type": "continuous",
                    "colors": _cmap_to_hex(cmap_name, 8),
                    "vmin": "0",
                    "vmax": "1",
                }

        # ── Argmax risk category (hidden) ───────────────────────────────
        if show_category:
            valid_mask = np.isfinite(probs[..., 0])
            category = np.full(probs.shape[:2], np.nan)
            category[valid_mask] = np.argmax(probs[valid_mask], axis=-1).astype(float)
            cat_url = _discrete_array_to_png_url(warp(category), n_states, overlay_opacity)
            layer_name = f"{node} — category"
            fg = folium.FeatureGroup(name=layer_name, show=False)
            folium.raster_layers.ImageOverlay(image=cat_url, bounds=bounds, opacity=1.0).add_to(fg)
            fg.add_to(m)
            layer_colorbars[layer_name] = {
                "title": layer_name,
                "type": "discrete",
                "colors": _DISCRETE_PALETTE_HEX[:n_states],
                "labels": list(states),
            }

        # ── Shannon entropy (hidden) ─────────────────────────────────────
        if show_entropy:
            ent = result.entropy(node)
            ent_max = math.log2(n_states) if n_states > 1 else 1.0
            ent_url = _array_to_png_url(warp(ent), "plasma", 0.0, ent_max, overlay_opacity)
            layer_name = f"{node} — entropy"
            fg = folium.FeatureGroup(name=layer_name, show=False)
            folium.raster_layers.ImageOverlay(image=ent_url, bounds=bounds, opacity=1.0).add_to(fg)
            fg.add_to(m)
            layer_colorbars[layer_name] = {
                "title": layer_name,
                "type": "continuous",
                "colors": _cmap_to_hex("plasma", 8),
                "vmin": "0",
                "vmax": f"{ent_max:.3g} bits",
            }

    # ── Extra layers ───────────────────────────────────────────────────────
    if extra_layers:
        for layer_name, arr in extra_layers.items():
            vmin = float(np.nanpercentile(arr, 2))
            vmax = float(np.nanpercentile(arr, 98))
            img_url = _array_to_png_url(warp(arr), "viridis", vmin, vmax, overlay_opacity)
            fg = folium.FeatureGroup(name=layer_name, show=False)
            folium.raster_layers.ImageOverlay(image=img_url, bounds=bounds, opacity=1.0).add_to(fg)
            fg.add_to(m)
            layer_colorbars[layer_name] = {
                "title": layer_name,
                "type": "continuous",
                "colors": _cmap_to_hex("viridis", 8),
                "vmin": f"{vmin:.4g}",
                "vmax": f"{vmax:.4g}",
            }

    folium.LayerControl(collapsed=False).add_to(m)

    # ── Radio buttons + colorbar UI ─────────────────────────────────────────
    # A single MacroElement handles both concerns:
    #   1. Make geobn's raster overlays mutually exclusive (radio buttons).
    #      Other overlays are left alone.
    #   2. Show a colorbar panel bottom-left for whichever raster is selected.
    from folium import MacroElement
    from jinja2 import Template

    colorbars_json = json.dumps(layer_colorbars)
    script_body = f"""
document.addEventListener('DOMContentLoaded', function () {{
    var layerColorbars = {colorbars_json};

    var mapDiv = document.querySelector('.folium-map');
    var cbDiv = document.createElement('div');
    cbDiv.id = 'geobn-colorbar';
    cbDiv.style.cssText = 'position:absolute;bottom:30px;left:10px;z-index:1000;'
        + 'background:rgba(255,255,255,0.85);padding:8px 12px;border-radius:6px;'
        + 'font-family:sans-serif;font-size:12px;min-width:160px;display:none;pointer-events:none;';
    mapDiv.appendChild(cbDiv);

    function updateColorbar(name) {{
        var spec = layerColorbars[name];
        if (!spec) {{ cbDiv.style.display = 'none'; return; }}
        var html = '<div style="font-weight:bold;margin-bottom:5px;">' + spec.title + '</div>';
        if (spec.type === 'continuous') {{
            html += '<div style="background:linear-gradient(to right,' + spec.colors.join(',')
                + ');height:14px;width:180px;border-radius:3px;"></div>';
            html += '<div style="display:flex;justify-content:space-between;width:180px;margin-top:3px;">'
                + '<span>' + spec.vmin + '</span><span>' + spec.vmax + '</span></div>';
        }} else {{
            html += '<div>';
            for (var i = 0; i < spec.colors.length; i++) {{
                html += '<div style="display:flex;align-items:center;gap:5px;margin-bottom:3px;">'
                    + '<div style="width:14px;height:14px;background:' + spec.colors[i]
                    + ';border-radius:2px;flex-shrink:0;"></div>'
                    + '<span>' + spec.labels[i] + '</span></div>';
            }}
            html += '</div>';
        }}
        cbDiv.innerHTML = html;
        cbDiv.style.display = 'block';
    }}

    function layerName(cb) {{
        var label = cb.closest('label') || cb.parentElement;
        var span = label ? label.querySelector('span') : null;
        return span ? span.textContent.trim() : '';
    }}

    var overlayInputs = Array.from(document.querySelectorAll(
        '.leaflet-control-layers-overlays input[type="checkbox"]'))
        .filter(function (cb) {{ return layerName(cb) in layerColorbars; }});
    overlayInputs.forEach(function (cb) {{
        cb.dataset.layerName = layerName(cb);

        cb.addEventListener('change', function () {{
            if (cb.checked) {{
                // Uncheck the other rasters so at most one is active at a time
                overlayInputs.forEach(function (other) {{
                    if (other !== cb && other.checked) {{
                        other.checked = false;
                        other.dispatchEvent(new Event('change'));
                    }}
                }});
                updateColorbar(cb.dataset.layerName);
            }} else {{
                // Active layer clicked again — deselect and hide colorbar
                cbDiv.style.display = 'none';
            }}
        }});
    }});
}});
"""
    ui_script = MacroElement()
    ui_script._template = Template(
        "{% macro script(this, kwargs) %}\n" + script_body + "\n{% endmacro %}"
    )
    ui_script.add_to(m)

    # ── Write HTML ─────────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / filename
    m.save(str(html_path))

    if open_browser:
        webbrowser.open(html_path.as_uri())

    return html_path
