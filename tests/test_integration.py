"""End-to-end integration tests using in-memory data (no file I/O needed)."""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

import geobn
from geobn.network import GeoBayesianNetwork


@pytest.fixture
def bn(fire_risk_model) -> GeoBayesianNetwork:
    return GeoBayesianNetwork(fire_risk_model)


class TestEndToEnd:
    def test_basic_infer(self, bn, slope_array, rainfall_array, reference_transform):
        bn.set_input(
            "slope",
            geobn.ArraySource(slope_array, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_input(
            "rainfall",
            geobn.ArraySource(rainfall_array, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        result = bn.infer(query=["fire_risk"])

        probs = result.probabilities["fire_risk"]
        assert probs.shape == (10, 10, 3)
        valid = ~np.isnan(probs[..., 0])
        np.testing.assert_allclose(probs[valid].sum(axis=-1), 1.0, atol=1e-5)

    def test_constant_source(self, bn, slope_array, reference_transform):
        """ConstantSource broadcasts correctly across the reference grid."""
        bn.set_input(
            "slope",
            geobn.ArraySource(slope_array, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_input("rainfall", geobn.ConstantSource(150.0))
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        result = bn.infer(query=["fire_risk"])
        probs = result.probabilities["fire_risk"]
        # All pixels should have the same rainfall evidence (high), so
        # probabilities should only vary with slope.
        assert not np.all(np.isnan(probs))

    def test_nodata_propagation(self, bn, slope_array, rainfall_array, reference_transform):
        """NaN in input → NaN in output for that pixel."""
        slope_with_nan = slope_array.copy()
        slope_with_nan[3, 3] = np.nan

        bn.set_input(
            "slope",
            geobn.ArraySource(slope_with_nan, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_input(
            "rainfall",
            geobn.ArraySource(rainfall_array, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        result = bn.infer(query=["fire_risk"])
        assert np.all(np.isnan(result.probabilities["fire_risk"][3, 3, :]))
        assert not np.any(np.isnan(result.probabilities["fire_risk"][0, 0, :]))

    def test_set_grid_override(self, bn, slope_array, reference_transform):
        """set_grid() overrides the reference grid derived from the first input."""
        # slope is 10×10 at 0.1°; override to 5×5 at 0.2°
        bn.set_input(
            "slope",
            geobn.ArraySource(slope_array, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_input("rainfall", geobn.ConstantSource(50.0))
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])
        bn.set_grid("EPSG:4326", 0.2, (0.0, 49.0, 1.0, 50.0))

        result = bn.infer(query=["fire_risk"])
        probs = result.probabilities["fire_risk"]
        assert probs.shape == (5, 5, 3)

    def test_missing_discretization_raises(self, bn, slope_array, reference_transform):
        bn.set_input(
            "slope",
            geobn.ArraySource(slope_array, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_input("rainfall", geobn.ConstantSource(50.0))
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        # rainfall discretization intentionally omitted

        with pytest.raises(ValueError, match="No discretization"):
            bn.infer(query=["fire_risk"])

    def test_set_input_non_root_raises(self, bn, slope_array, reference_transform):
        with pytest.raises(ValueError, match="parents"):
            bn.set_input(
                "fire_risk",
                geobn.ArraySource(slope_array, crs="EPSG:4326", transform=reference_transform),
            )

    def test_wrong_labels_raises(self, bn):
        with pytest.raises(ValueError, match="match"):
            bn.set_discretization("slope", [0, 10, 30, 90], ["a", "b", "WRONG"])

    def test_xarray_output(self, bn, slope_array, rainfall_array, reference_transform):
        bn.set_input(
            "slope",
            geobn.ArraySource(slope_array, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_input(
            "rainfall",
            geobn.ArraySource(rainfall_array, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        result = bn.infer(query=["fire_risk"])
        ds = result.to_xarray()

        assert "fire_risk" in ds
        assert "fire_risk_entropy" in ds
        assert ds["fire_risk"].dims == ("state", "y", "x")


class TestFetchRawAndSetInputArray:
    def test_fetch_raw_requires_grid(self, bn):
        with pytest.raises(RuntimeError, match="set_grid"):
            bn.fetch_raw(geobn.ConstantSource(1.0))

    def test_fetch_raw_returns_ndarray(self, bn):
        bn.set_grid("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        arr = bn.fetch_raw(geobn.ConstantSource(42.0))
        assert isinstance(arr, np.ndarray)
        assert arr.shape == bn._grid.shape
        assert np.all(arr == 42.0)

    def test_fetch_raw_passes_grid_to_source(self, bn, slope_array, reference_transform):
        bn.set_grid("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        source = geobn.ArraySource(slope_array, crs="EPSG:4326", transform=reference_transform)
        arr = bn.fetch_raw(source)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == bn._grid.shape

    def test_set_input_array_requires_grid(self, bn, slope_array):
        with pytest.raises(RuntimeError, match="set_grid"):
            bn.set_input_array("slope", slope_array)

    def test_set_input_array_registers_source(self, bn, slope_array):
        bn.set_grid("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        bn.set_input_array("slope", slope_array)
        assert "slope" in bn._inputs

    def test_set_input_array_end_to_end(self, bn, slope_array, rainfall_array):
        bn.set_grid("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        bn.set_input_array("slope", slope_array)
        bn.set_input_array("rainfall", rainfall_array)
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        result = bn.infer(query=["fire_risk"])
        probs = result.probabilities["fire_risk"]
        assert probs.shape == (10, 10, 3)
        valid = ~np.isnan(probs[..., 0])
        np.testing.assert_allclose(probs[valid].sum(axis=-1), 1.0, atol=1e-5)
