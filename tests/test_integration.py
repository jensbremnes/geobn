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

    def test_labels_inferred_from_bn(self, bn):
        """Labels omitted → read from pgmpy CPD state names."""
        bn.set_discretization("slope", [0, 10, 30, 90])
        spec = bn._discretizations["slope"]
        cpd = bn._model.get_cpds("slope")
        expected = list(cpd.state_names["slope"])
        assert spec.labels == expected

    def test_wrong_breakpoint_count_raises(self, bn):
        """Too few breakpoints for the number of BN states raises ValueError."""
        with pytest.raises(ValueError):
            bn.set_discretization("slope", [0, 10, 90])  # slope has 3 states → needs 4 breakpoints

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


class TestAutoGridResolution:
    """Auto-grid selects the finest-resolution self-contained source."""

    def test_finest_source_wins(self, bn, fire_risk_model):
        """Register coarse source first, then fine source; grid must match the fine one."""

        coarse_transform = Affine(0.5, 0, 0.0, 0, -0.5, 50.0)  # 0.5° pixels
        fine_transform = Affine(0.1, 0, 0.0, 0, -0.1, 50.0)    # 0.1° pixels (finer)
        coarse = np.ones((2, 2), dtype=np.float32) * 5.0
        fine = np.ones((10, 10), dtype=np.float32) * 5.0

        # Register coarse first — old code would pick this; new code should pick fine
        bn.set_input("slope",    geobn.ArraySource(coarse, crs="EPSG:4326", transform=coarse_transform))
        bn.set_input("rainfall", geobn.ArraySource(fine,   crs="EPSG:4326", transform=fine_transform))
        bn.set_discretization("slope",    [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        result = bn.infer(query=["fire_risk"])
        # Output grid shape should match the fine source (10×10), not the coarse (2×2)
        assert result.probabilities["fire_risk"].shape[:2] == (10, 10)

    def test_finest_source_wins_across_crs(self, bn):
        """A 10 m UTM grid beats a 0.001° (~79 m) WGS84 grid, although 0.001 < 10 in CRS units."""
        from pyproj import Transformer

        # 10 m UTM 32N raster around 9.01°E, 60.005°N
        x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform(9.0, 60.01)
        utm = np.ones((100, 100), dtype=np.float32) * 5.0
        utm_transform = Affine(10, 0, x0, 0, -10, y0)
        # 0.001° WGS84 raster covering the same area; registered last
        deg = np.ones((20, 20), dtype=np.float32) * 5.0
        deg_transform = Affine(0.001, 0, 9.0, 0, -0.001, 60.01)

        bn.set_input("slope",    geobn.ArraySource(utm, crs="EPSG:32632", transform=utm_transform))
        bn.set_input("rainfall", geobn.ArraySource(deg, crs="EPSG:4326",  transform=deg_transform))
        bn.set_discretization("slope",    [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        result = bn.infer(query=["fire_risk"])
        assert result.crs == "EPSG:32632"
        assert result.probabilities["fire_risk"].shape[:2] == (100, 100)

    def test_only_grid_aware_sources_raises(self, bn, fire_risk_model):
        """If every source requires_grid, auto-detection must raise a clear ValueError."""
        from geobn.sources._base import DataSource

        class FakeGridAwareSource(DataSource):
            requires_grid = True
            def fetch(self, grid=None):
                raise RuntimeError("should not be called")  # pragma: no cover

        bn.set_input("slope",    FakeGridAwareSource())
        bn.set_input("rainfall", FakeGridAwareSource())
        bn.set_discretization("slope",    [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        with pytest.raises(ValueError, match="set_grid"):
            bn.infer(query=["fire_risk"])

    def test_constant_only_source_raises(self, bn):
        """All-ConstantSource inputs have no CRS → must raise suggesting set_grid."""
        bn.set_input("slope",    geobn.ConstantSource(15.0))
        bn.set_input("rainfall", geobn.ConstantSource(50.0))
        bn.set_discretization("slope",    [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        with pytest.raises(ValueError, match="set_grid"):
            bn.infer(query=["fire_risk"])


class TestFetchRawAndArraySource:
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

    def test_array_source_no_crs_end_to_end(self, bn, slope_array, rainfall_array):
        """ArraySource(array) with no crs/transform is pre-aligned to the BN grid."""
        bn.set_grid("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        bn.set_input("slope",    geobn.ArraySource(slope_array))
        bn.set_input("rainfall", geobn.ArraySource(rainfall_array))
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        result = bn.infer(query=["fire_risk"])
        probs = result.probabilities["fire_risk"]
        assert probs.shape == (10, 10, 3)
        valid = ~np.isnan(probs[..., 0])
        np.testing.assert_allclose(probs[valid].sum(axis=-1), 1.0, atol=1e-5)

    def test_array_source_no_crs_wrong_shape_raises(self, bn, rainfall_array):
        bn.set_grid("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        bn.set_input("slope",    geobn.ArraySource(np.ones((4, 4), dtype=np.float32)))
        bn.set_input("rainfall", geobn.ArraySource(rainfall_array))
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        with pytest.raises(ValueError, match="without CRS"):
            bn.infer(query=["fire_risk"])

    def test_raster_source_nodata_gives_nan_posterior(
        self, bn, slope_array, reference_transform, tmp_path
    ):
        """Declared nodata in a GeoTIFF must yield NaN posteriors, not a binned sentinel."""
        import rasterio

        slope = slope_array.copy()
        slope[2, 7] = -9999.0
        path = tmp_path / "slope.tif"
        with rasterio.open(
            path, "w", driver="GTiff", height=10, width=10, count=1,
            dtype="float32", crs="EPSG:4326", transform=reference_transform, nodata=-9999.0,
        ) as dst:
            dst.write(slope, 1)

        bn.set_input("slope", geobn.RasterSource(path))
        bn.set_input("rainfall", geobn.ConstantSource(50.0))
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        probs = bn.infer(query=["fire_risk"]).probabilities["fire_risk"]
        assert np.all(np.isnan(probs[2, 7]))
        assert int(np.isnan(probs[..., 0]).sum()) == 1
