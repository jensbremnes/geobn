"""Interactive Leaflet map visualisation for InferenceResult.

Requires folium (``pip install geobn[viz]``).
"""
from __future__ import annotations

import base64
import math
import webbrowser
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .result import InferenceResult


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

    safe_range = vmax - vmin if vmax != vmin else 1.0
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

    _PALETTE = [
        (0.18, 0.80, 0.44),  # green  — low / state 0
        (0.90, 0.50, 0.13),  # orange — medium / state 1
        (0.91, 0.30, 0.24),  # red    — high / state 2
        (0.56, 0.28, 0.54),  # purple — state 3
        (0.20, 0.60, 0.86),  # blue   — state 4
    ]

    rgba = np.zeros((*category.shape, 4), dtype=np.float64)
    for i in range(min(n_states, len(_PALETTE))):
        r, g, b = _PALETTE[i]
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

    Returns
    -------
    Path
        Path to the written HTML file.
    """
    try:
        import folium
        import branca.colormap as branca_cm
    except ImportError as exc:
        raise ImportError(
            "folium is required for show_map(). "
            "Install it with: pip install geobn[viz]"
        ) from exc

    from pyproj import Transformer

    # ── WGS84 bounds from the result grid ─────────────────────────────────
    probs_any = next(iter(result.probabilities.values()))
    H, W = probs_any.shape[:2]
    t = result.transform
    x_min = t.c
    y_max = t.f
    x_max = x_min + W * t.a
    y_min = y_max + H * t.e  # t.e is negative

    crs_upper = (result.crs or "").upper()
    if crs_upper not in ("EPSG:4326", "WGS84", "CRS:84"):
        tr = Transformer.from_crs(result.crs, "EPSG:4326", always_xy=True)
        lon_min, lat_min = tr.transform(x_min, y_min)
        lon_max, lat_max = tr.transform(x_max, y_max)
    else:
        lon_min, lat_min = x_min, y_min
        lon_max, lat_max = x_max, y_max

    bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

    # ── Map + basemaps ─────────────────────────────────────────────────────
    m = folium.Map(location=center, zoom_start=9, control_scale=True)

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
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # ── Inference result layers ────────────────────────────────────────────
    for node, probs in result.probabilities.items():
        n_states = probs.shape[-1]
        states = result.state_names[node]

        # Probability band per state
        for i, state in enumerate(states):
            if i == 0:
                cmap_name = "YlGn"
            elif i == n_states - 1:
                cmap_name = "YlOrRd"
            else:
                cmap_name = "YlOrBr"

            shown = i == n_states - 1  # show last (highest-risk) state by default
            layer_name = f"P({state})"
            img_url = _array_to_png_url(
                probs[..., i], cmap_name, 0.0, 1.0, overlay_opacity
            )
            fg = folium.FeatureGroup(name=layer_name, show=shown)
            folium.raster_layers.ImageOverlay(image=img_url, bounds=bounds, opacity=1.0).add_to(fg)
            fg.add_to(m)

            if shown:
                cb = branca_cm.LinearColormap(
                    colors=_cmap_to_hex(cmap_name, 6),
                    vmin=0.0,
                    vmax=1.0,
                    caption=f"{node} — {layer_name}",
                )
                cb.add_to(m)

        # Argmax risk category
        valid_mask = np.isfinite(probs[..., 0])
        category = np.full(probs.shape[:2], np.nan)
        category[valid_mask] = np.argmax(probs[valid_mask], axis=-1).astype(float)
        cat_url = _discrete_array_to_png_url(category, n_states, overlay_opacity)
        fg = folium.FeatureGroup(name=f"{node} — category", show=False)
        folium.raster_layers.ImageOverlay(image=cat_url, bounds=bounds, opacity=1.0).add_to(fg)
        fg.add_to(m)

        # Shannon entropy
        ent = result.entropy(node)
        ent_max = math.log2(n_states) if n_states > 1 else 1.0
        ent_url = _array_to_png_url(ent, "plasma", 0.0, ent_max, overlay_opacity)
        fg = folium.FeatureGroup(name=f"{node} — entropy", show=False)
        folium.raster_layers.ImageOverlay(image=ent_url, bounds=bounds, opacity=1.0).add_to(fg)
        fg.add_to(m)

    # ── Extra layers ───────────────────────────────────────────────────────
    if extra_layers:
        for layer_name, arr in extra_layers.items():
            vmin = float(np.nanpercentile(arr, 2))
            vmax = float(np.nanpercentile(arr, 98))
            img_url = _array_to_png_url(arr, "viridis", vmin, vmax, overlay_opacity)
            fg = folium.FeatureGroup(name=layer_name, show=False)
            folium.raster_layers.ImageOverlay(image=img_url, bounds=bounds, opacity=1.0).add_to(fg)
            fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ── Write HTML ─────────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / filename
    m.save(str(html_path))

    if open_browser:
        webbrowser.open(html_path.as_uri())

    return html_path
