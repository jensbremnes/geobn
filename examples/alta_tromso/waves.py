"""Wave height in sheltered water from the offshore sea state, an approximation.

The wave model archive (WW3, 4 km) describes the open sea well but does not
resolve the sounds of the coast: a 4 km cell either has no value there or
spans water and islands alike.  This module brings the offshore sea state
onto the 250 m grid by how exposed each pixel is: a sheltered sound gets
small waves, the open sea keeps the offshore value.

Method
------
1. *Effective fetch* (U.S. Army Corps of Engineers, Shore Protection Manual,
   1984): from each water pixel, rays are cast upwind at 0, ±12, ±24 and ±36
   degrees from the wind direction until they meet land or reach
   ``max_fetch``.  Leaving the grid counts as open water.  The ray lengths
   are averaged with weights cos(angle).
2. *Exposure*: the offshore wave height is scaled by ``sqrt(F / F_open)``,
   capped at 1.  Fetch-limited waves grow with the square root of the fetch
   (JONSWAP, Hasselmann et al. 1973), so a sound with a fifth of the open
   fetch gets under half the offshore wave height.
3. *Local wind sea*: a long sound can raise its own waves in a strong wind,
   whatever the offshore sea does.  The JONSWAP fetch-limited height
   ``g Hs / U² = 0.0016 (g F / U²)^0.5`` (at most the fully developed
   ``0.243 U² / g``) is used where it is the larger of the two.

This ignores refraction, diffraction around headlands and wave-current
interaction, so treat it as an estimate of how exposed a pixel is, not as a
wave forecast.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

G = 9.81

# Ray angles either side of the wind direction, and their cos weights.
_RAY_OFFSETS_DEG = np.array([-36.0, -24.0, -12.0, 0.0, 12.0, 24.0, 36.0])


def effective_fetch(
    water: np.ndarray,
    wind_from_deg: float,
    pixel_size: float,
    max_fetch: float = 60_000.0,
) -> np.ndarray:
    """Effective upwind fetch in metres for every pixel of *water*.

    Parameters
    ----------
    water:
        (H, W) boolean, True for water.  Rows run north to south and columns
        west to east (a north-up grid).
    wind_from_deg:
        Direction the wind blows from, degrees clockwise from north.
    pixel_size:
        Pixel edge length in metres.
    max_fetch:
        Rays stop here, which also caps the fetch of the open sea.
    """
    h, w = water.shape
    rr, cc = np.nonzero(water)
    n_steps = int(max_fetch / pixel_size)
    total = np.zeros(rr.size)
    weights = np.cos(np.radians(_RAY_OFFSETS_DEG))

    for offset, weight in zip(_RAY_OFFSETS_DEG, weights):
        theta = np.radians(wind_from_deg + offset)
        # Upwind is the direction the wind comes from: north is -row.
        d_row, d_col = -np.cos(theta), np.sin(theta)
        length = np.full(rr.size, max_fetch)
        # Only the rays still over water are stepped further.
        active = np.arange(rr.size)
        for k in range(1, n_steps + 1):
            r = np.rint(rr[active] + k * d_row).astype(int)
            c = np.rint(cc[active] + k * d_col).astype(int)
            # Leaving the grid means open water from here on.
            inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            active, r, c = active[inside], r[inside], c[inside]
            hit = ~water[r, c]
            length[active[hit]] = k * pixel_size
            active = active[~hit]
            if active.size == 0:
                break
        total += weight * length

    fetch = np.full((h, w), np.nan)
    fetch[rr, cc] = total / weights.sum()
    return fetch


def fetch_limited_hs(fetch: np.ndarray, wind_speed: np.ndarray | float) -> np.ndarray:
    """JONSWAP significant wave height (m) for *fetch* (m) and *wind_speed* (m/s)."""
    u = np.maximum(np.asarray(wind_speed, dtype=float), 0.5)
    hs = 0.0016 * u**2 / G * np.sqrt(G * fetch / u**2)
    return np.minimum(hs, 0.243 * u**2 / G)


def sheltered_hs(
    offshore_hs: np.ndarray,
    fetch: np.ndarray,
    wind_speed: np.ndarray | float,
    open_fetch: float = 50_000.0,
) -> np.ndarray:
    """Wave height after scaling *offshore_hs* by exposure (see the module docstring)."""
    exposure = np.sqrt(np.clip(fetch / open_fetch, 0.0, 1.0))
    return np.fmax(offshore_hs * exposure, fetch_limited_hs(fetch, wind_speed))


def spread_offshore(hs: np.ndarray, sigma_px: float = 8.0) -> np.ndarray:
    """Fill the gaps of a coarse wave field with the nearest value and smooth it.

    The result is the offshore sea state next to every pixel, including the
    sounds the wave model has no values for.  The smoothing (a Gaussian of
    *sigma_px* pixels) removes the steps of the coarse model grid.
    """
    valid = np.isfinite(hs)
    if not valid.any():
        return np.full(hs.shape, np.nan)
    _, (ri, ci) = distance_transform_edt(~valid, return_indices=True)
    return gaussian_filter(hs[ri, ci], sigma_px)


def mean_direction(degrees: np.ndarray) -> float:
    """Circular mean of directions in degrees, ignoring NaN."""
    rad = np.radians(degrees[np.isfinite(degrees)])
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)
