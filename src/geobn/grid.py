"""Grid specification and pure-numpy/pyproj reprojection engine.

No rasterio objects are used here.  The only spatial library is pyproj,
which is used solely for coordinate transformation between CRS.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import Geod, Transformer
from pyproj.exceptions import ProjError

from ._types import RasterData

_log = logging.getLogger(__name__)

# Points sampled per grid edge in extent_wgs84 (same default as GDAL/rasterio)
_EXTENT_DENSIFY_PTS = 21

#: Resampling methods accepted by :func:`align_to_grid` and the data sources.
RESAMPLING_METHODS = ("bilinear", "nearest", "mode", "average", "min", "max")

# Upper bound on sub-samples per axis within one destination pixel for the
# aggregating methods.  A destination pixel spanning more source pixels than
# this is sampled on a 64 x 64 lattice instead of visiting every pixel.
_MAX_SUPERSAMPLE = 64

# Destination pixels probed per axis when estimating how many source pixels
# one destination pixel spans.
_FOOTPRINT_PROBES = 33

# Number of sub-samples held in memory at once by the aggregating methods.
_SAMPLE_BUDGET = 1 << 24


@dataclass
class GridSpec:
    """Describes the spatial grid that all inputs are aligned to."""

    crs: str                  # EPSG string or WKT
    transform: Affine         # pixel-corner-to-world affine (GDAL convention)
    shape: tuple[int, int]    # (height, width)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_raster_data(cls, data: RasterData) -> GridSpec:
        if data.crs is None or data.transform is None:
            raise ValueError(
                "Cannot derive a GridSpec from a source with no CRS/transform "
                "(e.g. ConstantSource).  Call bn.set_grid() explicitly."
            )
        return cls(crs=data.crs, transform=data.transform, shape=data.array.shape[:2])

    @classmethod
    def from_params(
        cls,
        crs: str,
        resolution: float,
        extent: tuple[float, float, float, float],
    ) -> GridSpec:
        """Build a GridSpec from explicit parameters.

        Parameters
        ----------
        crs:
            Target CRS as EPSG string (e.g. "EPSG:32632") or WKT.
        resolution:
            Pixel size in the units of *crs*.
        extent:
            (xmin, ymin, xmax, ymax) in the units of *crs*.
        """
        xmin, ymin, xmax, ymax = extent
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(
                f"extent must satisfy xmin < xmax and ymin < ymax, got {extent}"
            )
        width = max(1, round((xmax - xmin) / resolution))
        height = max(1, round((ymax - ymin) / resolution))
        # Affine(x_scale, x_skew, x_origin, y_skew, y_scale, y_origin)
        # y_scale is negative because raster rows increase downward.
        transform = Affine(resolution, 0, xmin, 0, -resolution, ymax)
        return cls(crs=crs, transform=transform, shape=(height, width))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def extent_wgs84(self) -> tuple[float, float, float, float]:
        """Return the bounding box in WGS84 (lon_min, lat_min, lon_max, lat_max).

        Points along all four grid edges are transformed, not just the
        corners, because straight edges in a projected CRS are curved in
        lon/lat.  If the grid contains a pole (e.g. polar stereographic), the
        box extends to that pole and spans all longitudes.  A grid crossing
        the antimeridian is not split; it gets a box spanning nearly
        -180..180 degrees longitude.
        """
        H, W = self.shape
        t = np.linspace(0.0, 1.0, _EXTENT_DENSIFY_PTS)
        # Pixel-space points along the top, right, bottom and left edges
        cols = np.concatenate([t * W, np.full_like(t, W), t * W, np.zeros_like(t)])
        rows = np.concatenate([np.zeros_like(t), t * H, np.full_like(t, H), t * H])
        xs, ys = self.transform * (cols, rows)
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(xs, ys)
        lon_min, lat_min = float(np.min(lons)), float(np.min(lats))
        lon_max, lat_max = float(np.max(lons)), float(np.max(lats))

        # A pole inside the grid is never reached by the edges
        inverse = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        for pole_lat in (90.0, -90.0):
            try:
                pole_x, pole_y = inverse.transform(0.0, pole_lat, errcheck=True)
            except ProjError:
                continue
            if not (np.isfinite(pole_x) and np.isfinite(pole_y)):
                continue
            col, row = ~self.transform * (pole_x, pole_y)
            if 0 <= col <= W and 0 <= row <= H:
                lon_min, lon_max = -180.0, 180.0
                if pole_lat > 0:
                    lat_max = 90.0
                else:
                    lat_min = -90.0
        return lon_min, lat_min, lon_max, lat_max


def _pixel_size_m(grid: GridSpec) -> float:
    """Return the ground size of the grid's centre pixel in metres.

    The pixel's column and row steps are measured as geodesic distances on
    the WGS84 ellipsoid, so grids in different CRSs (degrees, metres, feet)
    can be compared.  Non-square pixels are reduced to the geometric mean of
    the two sides, i.e. the side of a square with the same ground area.
    """
    H, W = grid.shape
    col, row = W / 2, H / 2
    xs, ys = zip(
        grid.transform * (col, row),
        grid.transform * (col + 1, row),
        grid.transform * (col, row + 1),
    )
    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)
    geod = Geod(ellps="WGS84")
    _, _, dx = geod.inv(lons[0], lats[0], lons[1], lats[1])
    _, _, dy = geod.inv(lons[0], lats[0], lons[2], lats[2])
    return float(np.sqrt(dx * dy))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align_to_grid(
    data: RasterData, grid: GridSpec, resampling: str = "bilinear"
) -> np.ndarray:
    """Reproject and resample *data* to match *grid*.

    Returns a (H, W) float32 array.  Pixels that fall outside *data*'s extent
    are filled with NaN.

    *resampling* is one of :data:`RESAMPLING_METHODS`:

    - ``"bilinear"`` interpolates between the four source pixels around each
      destination pixel centre.  A NaN among them gives NaN.
    - ``"nearest"`` takes the source pixel under the destination pixel centre.
    - ``"mode"``, ``"average"``, ``"min"`` and ``"max"`` aggregate over the
      source pixels that fall within each destination pixel, ignoring NaN.  A
      destination pixel is NaN only when none of them has data.  Where the
      source is coarser than the grid, a destination pixel lies within a
      single source pixel and these methods give the same result as
      ``"nearest"``.

    A source with ``crs=None`` must either be a single value (ConstantSource),
    which is broadcast, or already match the grid shape (pre-aligned
    ArraySource), which is returned as-is.  Any other ``crs=None`` shape raises
    ``ValueError``.  A source that already matches the target grid is returned
    as-is.  *resampling* does not apply to either case.
    """
    resampling = _check_resampling(resampling)
    if data.crs is None:
        if data.array.shape == grid.shape:
            # Pre-aligned array (ArraySource with no CRS) — return as-is
            _log.debug("Pre-aligned array — no reprojection needed")
            return data.array.astype(np.float32)
        if data.array.size != 1:
            raise ValueError(
                f"Array without CRS has shape {data.array.shape}, but the grid "
                f"shape is {grid.shape}.  Pass crs= and transform= to ArraySource "
                "so it can be reprojected, or supply an array already aligned to the grid."
            )
        # Scalar broadcast (ConstantSource)
        _log.debug("Broadcasting constant %g → shape %s", float(data.array.flat[0]), grid.shape)
        return np.full(grid.shape, float(data.array.flat[0]), dtype=np.float32)

    src_arr = data.array.astype(np.float32)

    if (
        data.crs == grid.crs
        and data.transform == grid.transform
        and data.array.shape == grid.shape
    ):
        _log.debug("Grid match — no reprojection needed")
        return src_arr

    _log.info("Reprojecting %s → %s (%s)", data.crs, grid.crs, resampling)
    return _reproject(
        src_arr, data.crs, data.transform, grid.crs, grid.transform, grid.shape,
        resampling=resampling,
    )


def _check_resampling(value: str) -> str:
    """Validate a ``resampling`` argument."""
    if not isinstance(value, str) or value not in RESAMPLING_METHODS:
        raise ValueError(
            "resampling must be one of "
            f"{', '.join(repr(m) for m in RESAMPLING_METHODS)}; got {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Core reprojection (pure numpy + pyproj)
# ---------------------------------------------------------------------------


def _reproject(
    src: np.ndarray,
    src_crs: str,
    src_transform: Affine,
    dst_crs: str,
    dst_transform: Affine,
    dst_shape: tuple[int, int],
    resampling: str = "bilinear",
) -> np.ndarray:
    H, W = dst_shape
    locate = _PixelMapper(src_crs, src_transform, dst_crs, dst_transform)

    if resampling in ("bilinear", "nearest"):
        # Pixel centre = corner + 0.5
        col_centre, row_centre = np.meshgrid(
            np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5
        )
        src_row, src_col = locate(row_centre, col_centre)
        if resampling == "bilinear":
            return _bilinear_resample(src, src_row, src_col)
        return _nearest_resample(src, src_row, src_col)

    return _aggregate_resample(src, locate, dst_shape, resampling)


class _PixelMapper:
    """Map destination pixel-grid coordinates to source pixel-grid coordinates."""

    def __init__(
        self, src_crs: str, src_transform: Affine, dst_crs: str, dst_transform: Affine
    ) -> None:
        self._dst_transform = dst_transform
        self._src_inv = ~src_transform
        self._transformer = (
            Transformer.from_crs(dst_crs, src_crs, always_xy=True)
            if dst_crs != src_crs
            else None
        )

    def __call__(
        self, dst_row: np.ndarray, dst_col: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        t = self._dst_transform
        # Destination pixel coordinates → world coordinates in the destination CRS
        dst_x = t.a * dst_col + t.b * dst_row + t.c
        dst_y = t.d * dst_col + t.e * dst_row + t.f

        # Destination CRS → source CRS
        if self._transformer is not None:
            shape = dst_x.shape
            src_x, src_y = self._transformer.transform(dst_x.ravel(), dst_y.ravel())
            src_x = np.asarray(src_x).reshape(shape)
            src_y = np.asarray(src_y).reshape(shape)
        else:
            src_x, src_y = dst_x, dst_y

        # Inverse source affine: world coords → fractional pixel indices
        inv = self._src_inv
        src_col = inv.a * src_x + inv.b * src_y + inv.c
        src_row = inv.d * src_x + inv.e * src_y + inv.f
        return src_row, src_col


def _bilinear_resample(
    src: np.ndarray,
    row_pix: np.ndarray,
    col_pix: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation at fractional pixel-grid coordinates.

    *row_pix* and *col_pix* are in pixel-grid space where 0.5 is the centre
    of the first pixel (GDAL / affine-package convention).
    """
    src_height, src_width = src.shape

    # Shift so that the centre of pixel 0 maps to coordinate 0.0
    row_centered = row_pix - 0.5
    col_centered = col_pix - 0.5

    # Integer indices of the four surrounding neighbours
    row_lo = np.floor(row_centered).astype(np.int32)
    col_lo = np.floor(col_centered).astype(np.int32)
    row_hi = row_lo + 1
    col_hi = col_lo + 1

    # Fractional distances from the lower neighbour (0–1)
    row_frac = (row_centered - row_lo).astype(np.float32)
    col_frac = (col_centered - col_lo).astype(np.float32)

    # Pixels outside source extent → NaN
    out_of_bounds = (
        (row_centered < -0.5) | (row_centered >= src_height - 0.5) |
        (col_centered < -0.5) | (col_centered >= src_width  - 0.5)
    )

    # Clamp indices for safe array access (out-of-bounds pixels are overwritten below)
    row_lo_safe = np.clip(row_lo, 0, src_height - 1)
    row_hi_safe = np.clip(row_hi, 0, src_height - 1)
    col_lo_safe = np.clip(col_lo, 0, src_width  - 1)
    col_hi_safe = np.clip(col_hi, 0, src_width  - 1)

    # Values at the four surrounding pixels (tl=top-left, tr=top-right, etc.)
    val_tl = src[row_lo_safe, col_lo_safe]
    val_tr = src[row_lo_safe, col_hi_safe]
    val_bl = src[row_hi_safe, col_lo_safe]
    val_br = src[row_hi_safe, col_hi_safe]

    # Bilinear interpolation: weighted average of the four neighbours
    result = (
        val_tl * (1 - row_frac) * (1 - col_frac)
        + val_tr * (1 - row_frac) * col_frac
        + val_bl * row_frac       * (1 - col_frac)
        + val_br * row_frac       * col_frac
    )
    n_out_of_bounds = int(out_of_bounds.sum())
    if n_out_of_bounds > 0:
        _log.debug("%d out-of-bounds pixel(s) set to NaN", n_out_of_bounds)
    result[out_of_bounds] = np.nan
    return result


def _nearest_resample(
    src: np.ndarray,
    row_pix: np.ndarray,
    col_pix: np.ndarray,
) -> np.ndarray:
    """Value of the source pixel containing each fractional pixel-grid coordinate.

    Coordinates outside the source extent, or not finite, give NaN.
    """
    src_height, src_width = src.shape
    with np.errstate(invalid="ignore"):
        inside = (
            (row_pix >= 0) & (row_pix < src_height)
            & (col_pix >= 0) & (col_pix < src_width)
        )
    rows = np.where(inside, row_pix, 0.0).astype(np.intp)  # truncation = floor for >= 0
    cols = np.where(inside, col_pix, 0.0).astype(np.intp)
    return np.where(inside, src[rows, cols], np.float32(np.nan)).astype(np.float32)


def _aggregate_resample(
    src: np.ndarray,
    locate: _PixelMapper,
    dst_shape: tuple[int, int],
    method: str,
) -> np.ndarray:
    """Aggregate the source pixels within each destination pixel.

    Each destination pixel is sampled on a regular k_row x k_col lattice of
    sub-pixel centres, fine enough that neighbouring samples are at most one
    source pixel apart, so every source pixel within the destination pixel is
    visited.  The samples are looked up with nearest-neighbour and reduced
    with *method*, ignoring NaN.

    Only the destination pixel corners are transformed to the source CRS.
    Sample positions are interpolated bilinearly between a pixel's four
    corners, which is exact for an affine relation and accurate to well
    within a source pixel for a projection over the span of one pixel.
    """
    H, W = dst_shape
    k_row, k_col = _supersample_factors(locate, dst_shape)
    n = k_row * k_col
    _log.debug("Aggregating (%s) over %d x %d samples per pixel", method, k_row, k_col)

    corner_col, corner_row = np.meshgrid(
        np.arange(W + 1, dtype=np.float64), np.arange(H + 1, dtype=np.float64)
    )
    corner_src_row, corner_src_col = locate(corner_row, corner_col)

    # Sub-pixel offsets within one destination pixel, each of shape (n,)
    sub_row, sub_col = np.meshgrid(
        (np.arange(k_row) + 0.5) / k_row, (np.arange(k_col) + 0.5) / k_col, indexing="ij"
    )
    # Bilinear weights of the four corners (top-left, top-right, bottom-left,
    # bottom-right) at each sub-sample
    weights = [
        ((1 - sub_row) * (1 - sub_col)).ravel(),
        ((1 - sub_row) * sub_col).ravel(),
        (sub_row * (1 - sub_col)).ravel(),
        (sub_row * sub_col).ravel(),
    ]

    def interpolate(corners: np.ndarray, r0: int, r1: int) -> np.ndarray:
        quad = (
            corners[r0:r1, :-1], corners[r0:r1, 1:],
            corners[r0 + 1:r1 + 1, :-1], corners[r0 + 1:r1 + 1, 1:],
        )
        return sum(c[:, :, None] * w for c, w in zip(quad, weights))

    result = np.empty((H, W), dtype=np.float32)
    rows_per_chunk = max(1, _SAMPLE_BUDGET // (W * n))
    for r0 in range(0, H, rows_per_chunk):
        r1 = min(H, r0 + rows_per_chunk)
        samples = _nearest_resample(
            src,
            interpolate(corner_src_row, r0, r1),
            interpolate(corner_src_col, r0, r1),
        )
        result[r0:r1] = _reduce(samples, method)
    return result


def _supersample_factors(
    locate: _PixelMapper, dst_shape: tuple[int, int]
) -> tuple[int, int]:
    """Sub-samples per axis needed to visit every source pixel in a destination pixel.

    A step of one destination pixel along each axis is measured in source
    pixels at a lattice of probe pixels across the grid, and the largest step
    sets the factor.  One sample per axis suffices where the source is at
    least as coarse as the grid.
    """
    H, W = dst_shape
    rows = np.unique(np.linspace(0, H - 1, min(H, _FOOTPRINT_PROBES)).round()) + 0.5
    cols = np.unique(np.linspace(0, W - 1, min(W, _FOOTPRINT_PROBES)).round()) + 0.5
    row, col = np.meshgrid(rows, cols, indexing="ij")
    r0, c0 = locate(row, col)
    r_down, c_down = locate(row + 1.0, col)
    r_right, c_right = locate(row, col + 1.0)
    # One destination step moves (dr, dc) in source pixels.  Keeping the
    # samples within one source pixel of each other along both source axes
    # takes as many sub-steps as the larger component.
    step_row = np.maximum(np.abs(r_down - r0), np.abs(c_down - c0))
    step_col = np.maximum(np.abs(r_right - r0), np.abs(c_right - c0))
    return _step_factor(step_row), _step_factor(step_col)


def _step_factor(step: np.ndarray) -> int:
    """Number of sub-steps that keeps every step at most one source pixel."""
    step = step[np.isfinite(step)]
    if step.size == 0:
        return 1
    k = int(np.ceil(float(step.max()) - 1e-9))
    if k > _MAX_SUPERSAMPLE:
        _log.debug(
            "A destination pixel spans %d source pixels per axis; sampling %d",
            k, _MAX_SUPERSAMPLE,
        )
    return min(max(k, 1), _MAX_SUPERSAMPLE)


def _reduce(samples: np.ndarray, method: str) -> np.ndarray:
    """Reduce the last axis of *samples* with *method*, ignoring NaN."""
    if method == "mode":
        return _nanmode(samples)
    reducer = {"average": np.nanmean, "min": np.nanmin, "max": np.nanmax}[method]
    with warnings.catch_warnings():
        # An all-NaN slice gives NaN, which is the intended result.
        warnings.simplefilter("ignore", RuntimeWarning)
        return reducer(samples, axis=-1).astype(np.float32)


def _nanmode(samples: np.ndarray) -> np.ndarray:
    """Most frequent non-NaN value along the last axis.

    Ties go to the smallest value.  A slice with no valid value gives NaN.
    """
    lead_shape = samples.shape[:-1]
    n = samples.shape[-1]
    ordered = np.sort(samples.reshape(-1, n), axis=-1)  # NaN sorts last
    position = np.arange(n)
    is_start = np.ones(ordered.shape, dtype=bool)
    is_start[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    run_start = np.maximum.accumulate(np.where(is_start, position, 0), axis=-1)
    # Length of the current run at each position.  Along a sorted row, the
    # first position to reach the longest run belongs to the smallest of the
    # tied values, and that is the position argmax returns.
    run_length = np.where(np.isnan(ordered), 0, position - run_start + 1)
    best = np.argmax(run_length, axis=-1)
    mode = ordered[np.arange(ordered.shape[0]), best]
    return mode.reshape(lead_shape).astype(np.float32)
