"""Grid specification and pure-numpy/pyproj reprojection engine.

No rasterio objects are used here.  The only spatial library is pyproj,
which is used solely for coordinate transformation between CRS.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_log = logging.getLogger(__name__)
from affine import Affine
from pyproj import Transformer

from ._types import RasterData


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
        """Return the bounding box in WGS84 (lon_min, lat_min, lon_max, lat_max)."""
        H, W = self.shape
        grid_transform = self.transform
        # Four corners in the source CRS
        corner_x = [
            grid_transform.c,
            grid_transform.c + W * grid_transform.a,
            grid_transform.c + H * grid_transform.b,
            grid_transform.c + W * grid_transform.a + H * grid_transform.b,
        ]
        corner_y = [
            grid_transform.f,
            grid_transform.f + W * grid_transform.d,
            grid_transform.f + H * grid_transform.e,
            grid_transform.f + W * grid_transform.d + H * grid_transform.e,
        ]
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(corner_x, corner_y)
        return float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align_to_grid(data: RasterData, grid: GridSpec) -> np.ndarray:
    """Reproject and resample *data* to match *grid*.

    Returns a (H, W) float32 array.  Pixels that fall outside *data*'s extent
    are filled with NaN.

    A source with ``crs=None`` (ConstantSource) is broadcast directly.
    A source that already matches the target grid is returned as-is.
    """
    if data.crs is None:
        if data.array.shape == grid.shape:
            # Pre-aligned array (ArraySource with no CRS) — return as-is
            _log.debug("Pre-aligned array — no reprojection needed")
            return data.array.astype(np.float32)
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

    _log.info("Reprojecting %s → %s", data.crs, grid.crs)
    return _reproject(src_arr, data.crs, data.transform, grid.crs, grid.transform, grid.shape)


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
) -> np.ndarray:
    H, W = dst_shape

    # Pixel-centre coordinates of every destination pixel
    col_idx = np.arange(W, dtype=np.float64)
    row_idx = np.arange(H, dtype=np.float64)
    col_grid, row_grid = np.meshgrid(col_idx, row_idx)

    # Pixel centre = corner + 0.5
    col_centre = col_grid + 0.5
    row_centre = row_grid + 0.5

    # Destination pixel centres in destination CRS (world coords)
    dst_x = dst_transform.a * col_centre + dst_transform.b * row_centre + dst_transform.c
    dst_y = dst_transform.d * col_centre + dst_transform.e * row_centre + dst_transform.f

    # Transform from destination CRS to source CRS
    if dst_crs != src_crs:
        crs_transformer = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
        src_x, src_y = crs_transformer.transform(dst_x.ravel(), dst_y.ravel())
        src_x = src_x.reshape(H, W)
        src_y = src_y.reshape(H, W)
    else:
        src_x, src_y = dst_x, dst_y

    # Apply the inverse source affine to convert world coords → fractional pixel indices
    src_affine_inv = ~src_transform
    src_col = src_affine_inv.a * src_x + src_affine_inv.b * src_y + src_affine_inv.c
    src_row = src_affine_inv.d * src_x + src_affine_inv.e * src_y + src_affine_inv.f

    return _bilinear_resample(src, src_row, src_col)


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
