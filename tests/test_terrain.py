"""Tests for geobn.terrain."""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pyproj import Geod, Transformer

import geobn
from geobn import terrain
from geobn.grid import GridSpec

# A 0.1 rise per metre: 5.71° or 10 %.
_RISE = 0.1
_SLOPE_DEG = float(np.degrees(np.arctan(_RISE)))


def _utm_grid(x0: float = 500_000.0, zone: int = 33) -> GridSpec:
    return GridSpec.from_params(f"EPSG:326{zone}", 10.0, (x0, 7_700_000.0, x0 + 400.0, 7_700_400.0))


def _plane_on_ground(grid: GridSpec, bearing: float) -> np.ndarray:
    """Elevation rising by _RISE per metre of ground distance toward *bearing*.

    Positions are measured geodesically from the grid's first pixel, so the
    plane is defined on the ground, not in CRS coordinates.
    """
    H, W = grid.shape
    cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    xs, ys = grid.transform * (cols, rows)
    lons, lats = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(xs, ys)
    az, _, dist = Geod(ellps="WGS84").inv(
        np.full(lons.shape, lons[0, 0]), np.full(lats.shape, lats[0, 0]), lons, lats
    )
    along = dist * np.cos(np.radians(np.asarray(az) - bearing))
    return (_RISE * along).astype(np.float64)


def _fetch(source, grid: GridSpec) -> np.ndarray:
    return source.fetch(grid=grid).array


class TestSlopeAndAspect:
    @pytest.mark.parametrize("rises_toward, faces", [(0, 180), (90, 270), (180, 0), (270, 90)])
    def test_plane_in_utm(self, rises_toward, faces):
        grid = _utm_grid()
        dem = geobn.ArraySource(_plane_on_ground(grid, rises_toward))

        slope = _fetch(terrain.slope(dem), grid)
        aspect = _fetch(terrain.aspect(dem), grid)

        np.testing.assert_allclose(slope, _SLOPE_DEG, atol=0.01)
        # Aspect wraps at north; compare on the circle.
        diff = (aspect - faces + 180) % 360 - 180
        np.testing.assert_allclose(diff, 0.0, atol=0.05)

    def test_plane_in_degrees_at_high_latitude(self):
        """Column spacing shrinks with latitude; the slope must not change across rows."""
        grid = GridSpec.from_params("EPSG:4326", 0.01, (20.0, 69.0, 20.2, 70.0))
        dem = geobn.ArraySource(_plane_on_ground(grid, 90.0))

        slope = _fetch(terrain.slope(dem), grid)

        np.testing.assert_allclose(slope, _SLOPE_DEG, atol=0.02)

    def test_aspect_is_from_true_north_off_the_central_meridian(self):
        """In UTM far from the central meridian, grid north is not true north."""
        grid = _utm_grid(x0=300_000.0)
        dem = geobn.ArraySource(_plane_on_ground(grid, 0.0))   # rises to true north

        aspect = _fetch(terrain.aspect(dem), grid)

        np.testing.assert_allclose(aspect, 180.0, atol=0.05)

        # The plane rising along grid north faces away from true south by the
        # meridian convergence, about 2° here.
        H, W = grid.shape
        rows = np.arange(H)[:, None] * np.ones((1, W))
        grid_north = geobn.ArraySource(-_RISE * 10.0 * rows)
        assert abs(float(np.mean(_fetch(terrain.aspect(grid_north), grid))) - 180.0) > 1.0

    def test_rotated_grid(self):
        """A grid rotated by 30° still gives true-north aspect."""
        angle = 30.0
        transform = Affine.translation(500_000.0, 7_700_000.0) * Affine.rotation(angle) * Affine.scale(10.0, -10.0)
        grid = GridSpec(crs="EPSG:32633", transform=transform, shape=(30, 30))
        dem = geobn.ArraySource(_plane_on_ground(grid, 90.0))   # rises to the east

        aspect = _fetch(terrain.aspect(dem), grid)
        slope = _fetch(terrain.slope(dem), grid)

        np.testing.assert_allclose(aspect, 270.0, atol=0.05)
        np.testing.assert_allclose(slope, _SLOPE_DEG, atol=0.01)

    def test_percent_units(self):
        grid = _utm_grid()
        dem = geobn.ArraySource(_plane_on_ground(grid, 90.0))

        np.testing.assert_allclose(_fetch(terrain.slope(dem, units="percent"), grid), 10.0, atol=0.01)

    def test_nan_edge_is_not_a_cliff(self):
        """Next to NaN the difference is one-sided, so the slope stays that of the plane."""
        grid = _utm_grid()
        z = _plane_on_ground(grid, 90.0)
        z[:, :10] = np.nan

        slope = _fetch(terrain.slope(geobn.ArraySource(z)), grid)

        assert np.isnan(slope[:, :10]).all()
        np.testing.assert_allclose(slope[:, 10:], _SLOPE_DEG, atol=0.01)

    def test_isolated_pixel_is_nan(self):
        grid = _utm_grid()
        z = np.full(grid.shape, np.nan)
        z[5, 5] = 100.0

        assert np.isnan(_fetch(terrain.slope(geobn.ArraySource(z)), grid)).all()

    def test_flat_aspect(self):
        grid = _utm_grid()
        dem = geobn.ArraySource(np.full(grid.shape, 100.0))

        assert np.isnan(_fetch(terrain.aspect(dem), grid)).all()
        np.testing.assert_array_equal(_fetch(terrain.aspect(dem, flat=-1.0), grid), -1.0)
        np.testing.assert_array_equal(_fetch(terrain.slope(dem), grid), 0.0)

    def test_matches_np_gradient_without_nan(self):
        grid = _utm_grid(x0=500_000.0 - 200.0)   # straddles the central meridian
        rng = np.random.default_rng(0)
        z = rng.normal(100.0, 5.0, grid.shape)

        slope = _fetch(terrain.slope(geobn.ArraySource(z)), grid)

        # UTM scale near the central meridian is 0.9996, so ground distance is
        # grid distance divided by it.
        dz_dy, dz_dx = np.gradient(z, 10.0 / 0.9996)
        expected = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
        np.testing.assert_allclose(slope, expected, rtol=1e-3, atol=1e-3)

    def test_invalid_arguments(self):
        dem = geobn.ConstantSource(1.0)
        with pytest.raises(ValueError, match="units"):
            terrain.slope(dem, units="radians")
        with pytest.raises(ValueError, match="flat must be a number"):
            terrain.aspect(dem, flat="north")


class TestWindows:
    grid = GridSpec.from_params("EPSG:32633", 1.0, (0.0, 0.0, 3.0, 3.0))
    z = np.arange(9.0).reshape(3, 3)

    def test_roughness(self):
        out = _fetch(terrain.roughness(geobn.ArraySource(self.z)), self.grid)

        # Centre window spans 0..8; the corner window holds 0, 1, 3, 4.
        assert out[1, 1] == 8.0
        assert out[0, 0] == 4.0

    def test_tpi(self):
        out = _fetch(terrain.tpi(geobn.ArraySource(self.z)), self.grid)

        assert out[1, 1] == pytest.approx(0.0)
        assert out[0, 0] == pytest.approx(0.0 - (1 + 3 + 4) / 3)

    def test_nan_neighbours_are_ignored(self):
        z = self.z.copy()
        z[0, 0] = np.nan

        rough = _fetch(terrain.roughness(geobn.ArraySource(z)), self.grid)
        tpi = _fetch(terrain.tpi(geobn.ArraySource(z)), self.grid)

        assert np.isnan(rough[0, 0]) and np.isnan(tpi[0, 0])
        assert rough[1, 1] == 7.0                                   # 1..8
        assert tpi[1, 1] == pytest.approx(4.0 - (1 + 2 + 3 + 5 + 6 + 7 + 8) / 7)

    def test_no_valid_neighbour_gives_nan_tpi(self):
        z = np.full((3, 3), np.nan)
        z[1, 1] = 5.0

        assert np.isnan(_fetch(terrain.tpi(geobn.ArraySource(z)), self.grid)).all()

    def test_wider_window(self):
        grid = GridSpec.from_params("EPSG:32633", 1.0, (0.0, 0.0, 5.0, 5.0))
        z = np.zeros((5, 5))
        z[2, 2] = 10.0

        tpi = _fetch(terrain.tpi(geobn.ArraySource(z), size=5), grid)

        assert tpi[2, 2] == pytest.approx(10.0)
        assert tpi[0, 0] == pytest.approx(-10.0 / 8)

    @pytest.mark.parametrize("size", [1, 2, 4, 3.0, True, "3"])
    def test_invalid_size(self, size):
        with pytest.raises(ValueError, match="odd integer"):
            terrain.roughness(geobn.ConstantSource(1.0), size=size)
