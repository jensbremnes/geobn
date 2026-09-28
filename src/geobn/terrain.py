"""Terrain derivatives computed from an elevation source.

Each function takes a source of elevations in metres and returns a
:class:`~geobn.DerivedSource`, which can be passed to
:meth:`~geobn.GeoBayesianNetwork.set_input` directly::

    dem = geobn.WCSSource(url, layer="dtm", cache_dir="cache")
    bn.set_input("slope_angle", geobn.terrain.slope(dem))
    bn.set_input("roughness",   geobn.terrain.roughness(dem))

The elevations are aligned to the reference grid first, so the derivatives
describe the terrain at the grid's resolution.

Slope and aspect use the ground distance between neighbouring pixel centres,
measured along the WGS84 ellipsoid, so they are correct on grids in degrees
and in projected CRSs alike.  Aspect is measured from true north, also where
the grid's columns do not point east, as in a UTM zone away from its central
meridian or on a polar grid.  Next to a NaN pixel the derivative uses a
one-sided difference, so the edge of the data (a coastline, the border of a
survey) does not appear as a cliff.

Roughness and TPI work on a square window of pixels and ignore NaN pixels
within it.
"""
from __future__ import annotations

import math

import numpy as np
from pyproj import Geod, Transformer

from .grid import GridSpec
from .sources._base import DataSource
from .sources.derived_source import DerivedSource, _GridDerivedSource

__all__ = ["slope", "aspect", "roughness", "tpi"]

_SLOPE_UNITS = ("degrees", "percent")

# Samples per axis at which the grid's geometry is measured exactly.
_LATTICE = 65

# Step, in pixels, over which the local scale and direction of an axis are measured.
_STEP = 1e-2


def slope(source: DataSource, units: str = "degrees") -> DerivedSource:
    """Slope of the terrain.

    Parameters
    ----------
    source:
        Elevation in metres.
    units:
        ``"degrees"`` (default), from 0 to 90, or ``"percent"``, the rise
        over the run times 100.

    Returns
    -------
    DerivedSource
        NaN where the elevation is NaN, or where a pixel has no valid
        neighbour along a row or a column.
    """
    if units not in _SLOPE_UNITS:
        raise ValueError(
            f"units must be one of {', '.join(repr(u) for u in _SLOPE_UNITS)}; "
            f"got {units!r}"
        )

    def _slope(z: np.ndarray, grid: GridSpec) -> np.ndarray:
        east, north = _gradient(z, grid)
        rise = np.hypot(east, north)
        if units == "percent":
            return 100.0 * rise
        return np.degrees(np.arctan(rise))

    return _GridDerivedSource(_slope, source)


def aspect(source: DataSource, flat: float = np.nan) -> DerivedSource:
    """Compass direction the terrain faces, i.e. of steepest descent.

    Parameters
    ----------
    source:
        Elevation in metres.
    flat:
        Value for pixels where the terrain is exactly level, which face no
        direction.  The default NaN treats them as NoData.  Pass a value
        outside 0–360, such as ``-1.0``, to give them a state of their own
        in the discretization.

    Returns
    -------
    DerivedSource
        Degrees clockwise from true north, from 0 up to 360: 0 faces north,
        90 east, 180 south and 270 west.  NaN where the elevation is NaN, or
        where a pixel has no valid neighbour along a row or a column.
    """
    flat = _check_number("flat", flat)

    def _aspect(z: np.ndarray, grid: GridSpec) -> np.ndarray:
        east, north = _gradient(z, grid)
        # The terrain faces down the gradient, hence the negated components.
        bearing = np.degrees(np.arctan2(-east, -north)) % 360.0
        bearing[(east == 0) & (north == 0)] = flat
        return bearing

    return _GridDerivedSource(_aspect, source)


def roughness(source: DataSource, size: int = 3) -> DerivedSource:
    """Largest elevation difference within a window around each pixel.

    This is the maximum minus the minimum elevation over a ``size`` × ``size``
    window centred on the pixel, the definition used by GDAL's
    ``gdaldem roughness`` (Wilson et al. 2007).

    Parameters
    ----------
    source:
        Elevation in metres.
    size:
        Window width in pixels: an odd integer, at least 3.

    Returns
    -------
    DerivedSource
        In the units of the elevation.  NaN where the elevation is NaN.
        NaN pixels within the window are ignored.
    """
    size = _check_size(size)

    def _roughness(z: np.ndarray) -> np.ndarray:
        high = np.full(z.shape, -np.inf)
        low = np.full(z.shape, np.inf)
        for window in _window_views(z, size):
            high = np.fmax(high, window)
            low = np.fmin(low, window)
        out = high - low
        out[np.isnan(z)] = np.nan
        return out

    return DerivedSource(_roughness, source)


def tpi(source: DataSource, size: int = 3) -> DerivedSource:
    """Topographic position index: elevation relative to its surroundings.

    This is the pixel's elevation minus the mean elevation of the other
    pixels in a ``size`` × ``size`` window around it, as in GDAL's
    ``gdaldem TPI`` (Weiss 2001).  Positive values mark ridges and hilltops,
    negative values valleys and hollows, and values near zero flat ground or
    an even slope.

    Parameters
    ----------
    source:
        Elevation in metres.
    size:
        Window width in pixels: an odd integer, at least 3.  A wider window
        picks out larger landforms.

    Returns
    -------
    DerivedSource
        In the units of the elevation.  NaN where the elevation is NaN or no
        other pixel in the window has data.  NaN pixels within the window are
        ignored.
    """
    size = _check_size(size)

    def _tpi(z: np.ndarray) -> np.ndarray:
        total = np.zeros(z.shape)
        count = np.zeros(z.shape)
        for window in _window_views(z, size):
            valid = ~np.isnan(window)
            total += np.where(valid, window, 0.0)
            count += valid
        # The centre pixel is in the window too; take it back out.
        centre_valid = ~np.isnan(z)
        total -= np.where(centre_valid, z, 0.0)
        count -= centre_valid
        with np.errstate(invalid="ignore", divide="ignore"):
            return z - total / count

    return DerivedSource(_tpi, source)


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------


def _gradient(z: np.ndarray, grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return the elevation gradient as (east, north) components in metres per metre."""
    z = z.astype(np.float64)
    H, W = z.shape
    if H < 2 or W < 2:
        nan = np.full(z.shape, np.nan)
        return nan, nan

    col_dist, row_dist, col_dir, row_dir = _axis_geometry(grid)
    along_cols = _axis_derivative(z, col_dist, axis=1)
    along_rows = _axis_derivative(z, row_dist, axis=0)

    # The two derivatives are the gradient projected onto the column and row
    # directions, u and v.  Solve g·u = along_cols, g·v = along_rows for g.
    (ue, un), (ve, vn) = col_dir, row_dir
    det = ue * vn - un * ve
    with np.errstate(invalid="ignore", divide="ignore"):
        east = (along_cols * vn - along_rows * un) / det
        north = (ue * along_rows - ve * along_cols) / det
    return east, north


def _axis_derivative(z: np.ndarray, dist: np.ndarray, axis: int) -> np.ndarray:
    """Derivative of *z* along *axis*, given the distances between neighbours.

    *dist* has one element fewer than *z* along *axis*.  The difference is
    central where both neighbours are valid, forward or backward where only
    one is, and NaN where neither is or the pixel itself is NaN.
    """
    pad = [(0, 0), (0, 0)]
    pad[axis] = (1, 1)
    zp = np.pad(z, pad, constant_values=np.nan)
    dp = np.pad(dist, pad, constant_values=np.nan)
    n = z.shape[axis]
    prev = np.take(zp, range(0, n), axis=axis)
    nxt = np.take(zp, range(2, n + 2), axis=axis)
    d_prev = np.take(dp, range(0, n), axis=axis)
    d_next = np.take(dp, range(1, n + 1), axis=axis)

    central = (nxt - prev) / (d_prev + d_next)
    forward = (nxt - z) / d_next
    backward = (z - prev) / d_prev
    out = np.where(
        np.isfinite(central), central,
        np.where(np.isfinite(forward), forward, backward),
    )
    out[np.isnan(z)] = np.nan
    return out


def _axis_geometry(grid: GridSpec):
    """Ground distances and directions of the grid's column and row axes.

    Returns ``(col_dist, row_dist, col_dir, row_dir)``: the geodesic distance
    in metres between horizontally adjacent pixel centres, shape (H, W-1),
    and between vertically adjacent ones, shape (H-1, W); and the unit
    (east, north) vectors of increasing column and increasing row at every
    pixel, each a pair of (H, W) arrays.

    The geometry changes slowly across a grid, so it is measured exactly on
    a lattice of at most ``_LATTICE`` × ``_LATTICE`` pixels and interpolated
    linearly in between.  Measuring every pixel would cost several seconds
    per million pixels for no visible difference.
    """
    H, W = grid.shape
    rows, cols = _lattice(H), _lattice(W)
    col_scale, row_scale, col_dir, row_dir = _local_geometry(grid, rows, cols)

    to_rows = _interp_matrix(np.arange(H) + 0.5, rows)
    to_cols = _interp_matrix(np.arange(W) + 0.5, cols)
    # A pair of neighbours is one pixel apart; take the scale half-way between them.
    col_dist = to_rows @ col_scale @ _interp_matrix(np.arange(1, W), cols).T
    row_dist = _interp_matrix(np.arange(1, H), rows) @ row_scale @ to_cols.T

    def _direction(east: np.ndarray, north: np.ndarray):
        east = to_rows @ east @ to_cols.T
        north = to_rows @ north @ to_cols.T
        norm = np.hypot(east, north)
        return east / norm, north / norm

    return col_dist, row_dist, _direction(*col_dir), _direction(*row_dir)


def _lattice(n: int) -> np.ndarray:
    """Pixel-centre coordinates of up to ``_LATTICE`` evenly spread samples along an axis."""
    return np.linspace(0.5, n - 0.5, min(n, _LATTICE))


def _interp_matrix(targets: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Matrix M such that ``M @ values`` interpolates *values* at *samples* linearly onto *targets*."""
    if len(samples) == 1:
        return np.ones((len(targets), 1))
    return np.stack(
        [np.interp(targets, samples, unit) for unit in np.eye(len(samples))], axis=1
    )


def _local_geometry(grid: GridSpec, rows: np.ndarray, cols: np.ndarray):
    """Scale (metres per pixel) and unit direction of both axes at the given pixel positions.

    Each is measured over a step of ``_STEP`` pixels, which is short enough
    for the geodesic to follow the grid axis.
    """
    cc, rr = np.meshgrid(cols, rows)
    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)

    def _lonlat(c: np.ndarray, r: np.ndarray):
        xs, ys = grid.transform * (c, r)
        return transformer.transform(xs, ys)

    lon, lat = _lonlat(cc, rr)
    geod = Geod(ellps="WGS84")
    col_az, _, col_step = geod.inv(lon, lat, *_lonlat(cc + _STEP, rr))
    row_az, _, row_step = geod.inv(lon, lat, *_lonlat(cc, rr + _STEP))
    return (
        np.asarray(col_step) / _STEP,
        np.asarray(row_step) / _STEP,
        _unit_vector(np.asarray(col_az)),
        _unit_vector(np.asarray(row_az)),
    )


def _unit_vector(azimuth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(east, north) components of a unit vector at *azimuth* degrees from north."""
    rad = np.radians(azimuth)
    return np.sin(rad), np.cos(rad)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def _window_views(z: np.ndarray, size: int):
    """Yield *z* shifted to each offset of a ``size`` × ``size`` window, NaN-padded."""
    r = size // 2
    zp = np.pad(z.astype(np.float64), r, constant_values=np.nan)
    H, W = z.shape
    for dr in range(size):
        for dc in range(size):
            yield zp[dr:dr + H, dc:dc + W]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_size(size: int) -> int:
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 3 or size % 2 == 0:
        raise ValueError(f"size must be an odd integer of at least 3; got {size!r}")
    return int(size)


def _check_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a number; got {value!r}")
    value = float(value)
    if math.isinf(value):
        raise ValueError(f"{name} must be finite or NaN; got {value!r}")
    return value
