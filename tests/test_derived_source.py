"""Tests for DerivedSource."""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

import geobn
from geobn._types import RasterData
from geobn.grid import GridSpec
from geobn.sources import PointGridSource


@pytest.fixture
def small_grid() -> GridSpec:
    """5×5 grid over a small area in WGS84."""
    return GridSpec(crs="EPSG:4326", transform=Affine(0.1, 0, 5.0, 0, -0.1, 62.0), shape=(5, 5))


@pytest.fixture
def bn(fire_risk_model):
    return geobn.GeoBayesianNetwork(fire_risk_model)


class _CountingSource(geobn.sources.DataSource):
    """A constant that records how many times it was fetched."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value
        self.calls = 0

    def _fetch(self, grid=None) -> RasterData:
        self.calls += 1
        return RasterData(np.array([[self._value]], dtype=np.float32), None, None)


class TestDerivedSource:
    def test_applies_fn_to_one_source(self, small_grid):
        values = np.arange(25, dtype=np.float32).reshape(5, 5)
        derived = geobn.DerivedSource(lambda z: z * 2, geobn.ArraySource(values))

        data = derived.fetch(grid=small_grid)

        np.testing.assert_array_equal(data.array, values * 2)

    def test_passes_sources_in_order(self, small_grid):
        derived = geobn.DerivedSource(
            lambda a, b: a - b, geobn.ConstantSource(10.0), geobn.ConstantSource(3.0)
        )

        np.testing.assert_array_equal(derived.fetch(grid=small_grid).array, np.full((5, 5), 7.0))

    def test_output_sits_on_the_reference_grid(self, small_grid):
        data = geobn.DerivedSource(lambda z: z, geobn.ConstantSource(1.0)).fetch(grid=small_grid)

        assert data.array.dtype == np.float32
        assert data.crs == small_grid.crs
        assert data.transform == small_grid.transform

    def test_inputs_are_aligned_before_fn(self, small_grid):
        """A coarser input in the same CRS is resampled onto the grid before fn sees it."""
        seen = []
        coarse = geobn.ArraySource(
            np.full((2, 2), 4.0, dtype=np.float32),
            crs="EPSG:4326",
            transform=Affine(0.25, 0, 5.0, 0, -0.25, 62.0),
        )

        def fn(z):
            seen.append(z.shape)
            return z + 1

        data = geobn.DerivedSource(fn, coarse).fetch(grid=small_grid)

        assert seen == [(5, 5)]
        np.testing.assert_allclose(data.array, 5.0)

    def test_nan_passes_through(self, small_grid):
        values = np.ones((5, 5), dtype=np.float32)
        values[0, 0] = np.nan

        data = geobn.DerivedSource(lambda z: z + 1, geobn.ArraySource(values)).fetch(grid=small_grid)

        assert np.isnan(data.array[0, 0])
        assert data.array[1, 1] == 2.0

    def test_wrong_output_shape_raises(self, small_grid):
        def total(z):
            return z.sum()

        derived = geobn.DerivedSource(total, geobn.ConstantSource(1.0))

        with pytest.raises(ValueError, match=r"'.*total' returned an array of shape \(\)"):
            derived.fetch(grid=small_grid)

    def test_valid_range_applies_to_the_result(self, small_grid):
        derived = geobn.DerivedSource(
            lambda z: -z, geobn.ConstantSource(5.0), valid_range=(0.0, None)
        )

        assert np.isnan(derived.fetch(grid=small_grid).array).all()

    def test_nests(self, small_grid):
        inner = geobn.DerivedSource(lambda z: z + 1, geobn.ConstantSource(1.0))
        outer = geobn.DerivedSource(lambda a, b: a * b, inner, geobn.ConstantSource(3.0))

        np.testing.assert_array_equal(outer.fetch(grid=small_grid).array, np.full((5, 5), 6.0))

    def test_inputs_are_fetched_on_every_fetch(self, small_grid):
        source = _CountingSource(1.0)
        derived = geobn.DerivedSource(lambda z: z, source)
        assert source.calls == 0          # constructing fetches nothing

        derived.fetch(grid=small_grid)
        derived.fetch(grid=small_grid)

        assert source.calls == 2

    @pytest.mark.parametrize(
        "args, match",
        [
            (("not callable", geobn.ConstantSource(1.0)), "fn must be a function"),
            ((lambda z: z,), "at least one input source"),
            ((lambda z: z, np.ones((2, 2))), "must be a data source"),
        ],
    )
    def test_invalid_arguments_raise(self, args, match):
        with pytest.raises(ValueError, match=match):
            geobn.DerivedSource(*args)

    def test_requires_grid_follows_the_sources(self):
        point = PointGridSource(fn=lambda lat, lon: 1.0, sample_points=2, delay=0.0)

        assert geobn.DerivedSource(lambda z: z, point).requires_grid
        assert not geobn.DerivedSource(
            lambda a, b: a, point, geobn.ConstantSource(1.0)
        ).requires_grid

    def test_grid_aware_inputs_without_grid_raise(self):
        point = PointGridSource(fn=lambda lat, lon: 1.0, sample_points=2, delay=0.0)
        with pytest.raises(ValueError, match="requires a grid context"):
            geobn.DerivedSource(lambda z: z, point).fetch()

    def test_blind_fetch_probes_the_first_self_contained_input(self):
        transform = Affine(0.1, 0, 5.0, 0, -0.1, 62.0)
        derived = geobn.DerivedSource(
            lambda a, b: a + b,
            geobn.ConstantSource(1.0),
            geobn.ArraySource(np.ones((4, 4), dtype=np.float32), crs="EPSG:4326", transform=transform),
        )

        # The constant comes first but carries no CRS; the probe still returns it,
        # and infer() moves on to the next candidate for the grid.
        data = derived.fetch()

        assert data.crs is None


class TestDerivedSourceInNetwork:
    def test_infer_uses_the_derived_values(self, bn):
        bn.set_grid("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        # 20 − 15 = 5 → "flat" everywhere.
        bn.set_input(
            "slope",
            geobn.DerivedSource(lambda a, b: a - b, geobn.ConstantSource(20.0), geobn.ConstantSource(15.0)),
        )
        bn.set_input("rainfall", geobn.ConstantSource(50.0))
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        derived = bn.infer(query=["fire_risk"]).probabilities["fire_risk"]

        bn.set_input("slope", geobn.ConstantSource(5.0))
        direct = bn.infer(query=["fire_risk"]).probabilities["fire_risk"]
        np.testing.assert_array_equal(derived, direct)

    def test_derived_values_are_used_after_the_auto_grid_probe(self, bn):
        """The probe seeds the grid, but the node gets fn's result, not the raw input."""
        transform = Affine(0.1, 0, 0.0, 0, -0.1, 50.0)
        raw = geobn.ArraySource(
            np.full((10, 10), 50.0, dtype=np.float32), crs="EPSG:4326", transform=transform
        )
        # Raw 50 would be "steep"; the derived 5 is "flat".
        bn.set_input("slope", geobn.DerivedSource(lambda z: z / 10, raw))
        bn.set_input("rainfall", geobn.ConstantSource(50.0))
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        result = bn.infer(query=["fire_risk"])

        assert result.probabilities["fire_risk"].shape == (10, 10, 3)
        np.testing.assert_allclose(result.probabilities["fire_risk"][0, 0], [0.6, 0.3, 0.1], atol=1e-6)

    def test_frozen_derived_input_is_not_refetched(self, bn):
        bn.set_grid("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        source = _CountingSource(5.0)
        bn.set_input("slope", geobn.DerivedSource(lambda z: z, source))
        bn.set_input("rainfall", geobn.ConstantSource(50.0))
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])
        bn.freeze("slope")

        bn.infer(query=["fire_risk"])
        bn.infer(query=["fire_risk"])

        assert source.calls == 1

    def test_fetch_raw_returns_the_derived_values(self, bn):
        bn.set_grid("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))

        values = bn.fetch_raw(geobn.DerivedSource(lambda z: z + 1, geobn.ConstantSource(1.0)))

        np.testing.assert_array_equal(values, np.full((10, 10), 2.0))
