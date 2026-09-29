"""Figures for the Alta → Tromsø example: the scenario map and the timelapse."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
LAND = np.array([0.96, 0.95, 0.92])
SEA_SHALLOW = np.array([0.80, 0.88, 0.96])
SEA_DEEP = np.array([0.56, 0.71, 0.88])
WATER_LABEL = "#27496d"

# One hue for one magnitude: passage risk, light to dark orange-red.
RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk", ["#fde2c8", "#f7a263", "#e0622b", "#b23a12", "#7a2006"]
)
RISK_NORM = Normalize(0.0, 1.0)

HALO = [pe.withStroke(linewidth=3, foreground="white")]

# Where each place name sits relative to its point: (row offset, column
# offset, horizontal alignment).  The default is centred just above it.
LABEL_OFFSETS = {
    "Lopphavet": (0, 0, "center"),
    "Loppa": (-6, 12, "left"),
    "Alta": (6, 10, "left"),
    "Tromsø": (-8, -8, "right"),
}


def base_map(elevation: np.ndarray, pixel_size: float = 250.0) -> np.ndarray:
    """RGB image: hillshaded land, sea shaded from light (shallow) to blue (deep)."""
    land = elevation > 0
    ls = LightSource(azdeg=315, altdeg=40)
    shade = ls.hillshade(np.where(land, elevation, 0.0), vert_exag=2.0,
                         dx=pixel_size, dy=pixel_size)
    rgb = np.empty(elevation.shape + (3,))
    rgb[land] = LAND * (0.72 + 0.28 * shade[land, None])
    depth = np.clip(-elevation / 400.0, 0, 1)[..., None]
    sea = SEA_SHALLOW * (1 - depth) + SEA_DEEP * depth
    rgb[~land] = sea[~land]
    return np.clip(rgb, 0, 1)


def risk_rgba(risk: np.ndarray) -> np.ndarray:
    """Risk as colour with opacity growing with risk; low risk stays see-through."""
    rgba = RISK_CMAP(RISK_NORM(np.nan_to_num(risk)))
    alpha = np.clip((np.nan_to_num(risk) - 0.08) / 0.3, 0, 1) * 0.85
    rgba[..., 3] = np.where(np.isfinite(risk), alpha, 0)
    return rgba


def _draw_route(ax, path, style):
    ax.plot(path[:, 1], path[:, 0], **style)


SHORTEST = dict(color=INK_2, lw=1.5, ls=(0, (4, 3)), solid_capstyle="round",
                path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])
RISK_AWARE = dict(color=INK, lw=2.4, solid_capstyle="round",
                  path_effects=[pe.withStroke(linewidth=5, foreground="white")])


def _decorate(ax, ends, places, fontsize=10):
    for name, (r, c) in places.items():
        water = name == "Lopphavet"
        dr, dc, ha = LABEL_OFFSETS.get(name, (-8, 0, "center"))
        ax.text(c + dc, r + dr, name, ha=ha, va="center" if water else "bottom",
                fontsize=fontsize + (1 if water else 0),
                style="italic" if water else "normal",
                color=WATER_LABEL if water else INK,
                path_effects=HALO, zorder=6,
                transform=ax.transData)
    for rc in ends.values():
        ax.plot(rc[1], rc[0], "o", ms=7, mfc="white", mec=INK, mew=1.8, zorder=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")


def hero_figure(path: Path, elevation, results: dict, ends: dict, places: dict) -> Path:
    """Two panels, calm and storm, on the same extent and colour scale."""
    base = base_map(elevation)
    h, w = elevation.shape
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(15, 6.1), dpi=160)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.13, wspace=0.03)

    for ax, name in zip(axes, names):
        res = results[name]
        sc = res["scenario"]
        ax.imshow(base, interpolation="bilinear")
        ax.imshow(risk_rgba(res["risk"]), interpolation="nearest")
        routes = res["routes"]
        _draw_route(ax, routes["shortest"].path, SHORTEST)
        _draw_route(ax, routes["risk_aware"].path, RISK_AWARE)
        _decorate(ax, ends, places)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(f"{sc.label} · {sc.time.strftime('%d %b %Y').lstrip('0')}, "
                     f"{sc.time:%H:%M} UTC",
                     loc="left", fontsize=12.5, color=INK, pad=6)

        ra, sh = routes["risk_aware"], routes["shortest"]
        extra = ra.length_km - sh.length_km
        text = (f"Route: {ra.length_km:.0f} km, {risk_word(ra.mean_risk)} risk\n"
                f"Shortest: {sh.length_km:.0f} km, {risk_word(sh.mean_risk)} risk")
        if extra >= 1:
            text += f"\nDetour for safety: +{extra:.0f} km"
        ax.text(0.985, 0.03, text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10.5, color=INK, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#c3c2b7", lw=0.8))

    # Shared legend and colour scale.
    handles = [
        Line2D([], [], **{k: v for k, v in RISK_AWARE.items() if k != "path_effects"},
               label="Risk-aware route"),
        Line2D([], [], **{k: v for k, v in SHORTEST.items() if k != "path_effects"},
               label="Shortest route"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.015),
               ncol=2, frameon=False, fontsize=10.5)
    cax = fig.add_axes([0.62, 0.065, 0.3, 0.022])
    sm = plt.cm.ScalarMappable(norm=RISK_NORM, cmap=RISK_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["low", "medium", "high"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9.5, colors=INK_2, length=0)
    cax.set_title("Passage risk for the vessel (from geobn)", fontsize=10, color=INK_2,
                  loc="left", pad=4)

    fig.suptitle("A robot boat from Alta to Tromsø: the same trip on a calm day and in a storm",
                 x=0.01, ha="left", fontsize=15, color=INK, y=0.985)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def timelapse(path: Path, elevation, frames: list, ends: dict, places: dict) -> Path:
    """GIF of the storm arriving, with the route re-planned for every frame."""
    base = base_map(elevation)
    h, w = elevation.shape
    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=100)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01)
    def draw(i):
        t, risk, routes = frames[i]
        ax.clear()
        ax.imshow(base, interpolation="bilinear")
        ax.imshow(risk_rgba(risk), interpolation="nearest")
        _draw_route(ax, routes["shortest"].path, SHORTEST)
        _draw_route(ax, routes["risk_aware"].path, RISK_AWARE)
        _decorate(ax, ends, places, fontsize=9)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ra = routes["risk_aware"]
        ax.set_title(f"{t:%d %b %Y %H:%M} UTC · route {ra.length_km:.0f} km, "
                     f"{risk_word(ra.mean_risk)} risk", loc="left", fontsize=11, color=INK)

    anim = FuncAnimation(fig, draw, frames=len(frames))
    # pgmpy sets the root logger to INFO, which would print the writer's notice.
    logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)
    anim.save(path, writer=PillowWriter(fps=2))
    plt.close(fig)
    return path


def risk_word(mean_risk: float) -> str:
    """Name the expected risk of a route: below 0.2 low, 0.2–0.5 medium, above high."""
    return "low" if mean_risk < 0.2 else "medium" if mean_risk <= 0.5 else "high"
