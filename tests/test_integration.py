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

    def test_out_of_range_nan(self, bn, slope_array, rainfall_array, reference_transform):
        """out_of_range="nan" → NaN only where the value is outside the breakpoints."""
        slope = slope_array.copy()
        slope[2, 5] = 120.0  # above the last breakpoint (90)
        bn.set_input(
            "slope",
            geobn.ArraySource(slope, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_input(
            "rainfall",
            geobn.ArraySource(rainfall_array, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_discretization(
            "slope", [0, 10, 30, 90], ["flat", "moderate", "steep"], out_of_range="nan"
        )
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        probs = bn.infer(query=["fire_risk"]).probabilities["fire_risk"]
        nan_pixels = np.isnan(probs[..., 0])
        assert nan_pixels[2, 5]
        assert nan_pixels.sum() == 1

    def test_out_of_range_clip_is_default(self, bn, slope_array, reference_transform):
        slope = slope_array.copy()
        slope[2, 5] = 120.0
        bn.set_input(
            "slope",
            geobn.ArraySource(slope, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_input("rainfall", geobn.ConstantSource(50.0))
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        probs = bn.infer(query=["fire_risk"]).probabilities["fire_risk"]
        assert not np.isnan(probs).any()

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

    def test_set_input_non_root_runs_backwards(self, bn, reference_transform):
        """A source on a child node gives a posterior over its parent."""
        from pgmpy.inference import VariableElimination

        # fire_risk is a child of slope and rainfall; observe it and ask for slope.
        observed = np.tile([0.0, 1.0, 2.0, 1.0, 0.0], (10, 2)).astype(np.float32)
        bn.set_input(
            "fire_risk",
            geobn.ArraySource(observed, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_discretization("fire_risk", [-0.5, 0.5, 1.5, 2.5], ["low", "medium", "high"])

        probs = bn.infer(query=["slope"]).probabilities["slope"]
        assert probs.shape == (10, 10, 3)
        assert not np.isnan(probs).any()
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)

        ve = VariableElimination(bn._model)
        for value, state in [(0.0, "low"), (1.0, "medium"), (2.0, "high")]:
            expected = ve.query(
                ["slope"], evidence={"fire_risk": state}, show_progress=False
            ).values
            row, col = np.argwhere(observed == value)[0]
            np.testing.assert_allclose(probs[row, col], expected, atol=1e-6)

        # The posterior must actually differ from the prior, or the test proves nothing.
        prior = ve.query(["slope"], show_progress=False).values
        assert not np.allclose(probs[0, 0], prior, atol=1e-3)

    def test_mixed_root_and_non_root_evidence(self, bn, slope_array, reference_transform):
        """A root input and a child input together match a direct pgmpy query."""
        from pgmpy.inference import VariableElimination

        observed = np.full((10, 10), 2.0, dtype=np.float32)  # fire_risk = high everywhere
        bn.set_input(
            "slope",
            geobn.ArraySource(slope_array, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_input(
            "fire_risk",
            geobn.ArraySource(observed, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("fire_risk", [-0.5, 0.5, 1.5, 2.5], ["low", "medium", "high"])

        probs = bn.infer(query=["rainfall"]).probabilities["rainfall"]
        ve = VariableElimination(bn._model)
        expected = ve.query(
            ["rainfall"],
            evidence={"slope": "flat", "fire_risk": "high"},
            show_progress=False,
        ).values
        np.testing.assert_allclose(probs[0, 0], expected, atol=1e-6)

    def test_query_node_that_is_also_input_raises(self, bn, slope_array, reference_transform):
        """The guard fires before any data is fetched."""
        bn.set_input(
            "slope",
            geobn.ArraySource(slope_array, crs="EPSG:4326", transform=reference_transform),
        )
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        # A source that would blow up on fetch: if the guard ran late, we would see
        # a file error from here instead of the ValueError below.
        bn.set_input("rainfall", geobn.RasterSource("does_not_exist.tif"))
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        with pytest.raises(ValueError, match="both an input and a query node"):
            bn.infer(query=["slope"])
        with pytest.raises(ValueError, match="both an input and a query node"):
            bn.precompute(query=["slope"])

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
            def _fetch(self, grid=None):
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


class TestSuggestBreakpoints:
    def setup_method(self):
        self.grid = ("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))

    def test_n_defaults_to_the_bn_state_count(self, bn, slope_array):
        bn.set_grid(*self.grid)
        bn.set_input("slope", geobn.ArraySource(slope_array))
        # "slope" has three states in the fixture BN
        assert len(bn.suggest_breakpoints("slope")) == 4

    def test_result_round_trips_into_set_discretization(self, bn, slope_array):
        bn.set_grid(*self.grid)
        bn.set_input("slope", geobn.ArraySource(slope_array))
        bn.set_discretization("slope", bn.suggest_breakpoints("slope"))
        spec = bn._discretizations["slope"]
        assert spec.labels == ["flat", "moderate", "steep"]

    def test_quantile_fills_every_state(self, bn, rainfall_array):
        bn.set_grid(*self.grid)
        bn.set_input("rainfall", geobn.ArraySource(rainfall_array))
        bn.set_discretization("rainfall", bn.suggest_breakpoints("rainfall"))
        bn.set_input("slope", geobn.ConstantSource(5.0))
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])

        result = bn.infer(query=["fire_risk"])
        assert result.probabilities["fire_risk"].shape == (10, 10, 3)

    def test_equal_interval_method(self, bn, slope_array):
        bn.set_grid(*self.grid)
        bn.set_input("slope", geobn.ArraySource(slope_array))
        bp = bn.suggest_breakpoints("slope", method="equal_interval")
        widths = np.diff(bp)
        np.testing.assert_allclose(widths, widths[0])

    def test_explicit_n_overrides_the_state_count(self, bn, slope_array):
        bn.set_grid(*self.grid)
        bn.set_input("slope", geobn.ArraySource(slope_array))
        assert len(bn.suggest_breakpoints("slope", n=5)) == 6

    def test_bounds_are_passed_through(self, bn, slope_array):
        bn.set_grid(*self.grid)
        bn.set_input("slope", geobn.ArraySource(slope_array))
        bp = bn.suggest_breakpoints("slope", bounds=(0.0, 90.0))
        assert bp[0] == 0.0
        assert bp[-1] == 90.0

    def test_requires_a_grid(self, bn, slope_array):
        bn.set_input("slope", geobn.ArraySource(slope_array))
        with pytest.raises(RuntimeError, match="set_grid"):
            bn.suggest_breakpoints("slope")

    def test_unregistered_node_raises(self, bn):
        bn.set_grid(*self.grid)
        with pytest.raises(ValueError, match="set_input"):
            bn.suggest_breakpoints("slope")

    def test_unknown_node_raises(self, bn):
        bn.set_grid(*self.grid)
        with pytest.raises(ValueError, match="does not exist"):
            bn.suggest_breakpoints("humidity")

    def test_unknown_method_raises(self, bn, slope_array):
        bn.set_grid(*self.grid)
        bn.set_input("slope", geobn.ArraySource(slope_array))
        with pytest.raises(ValueError, match="Unknown method"):
            bn.suggest_breakpoints("slope", method="jenks")

    def test_method_is_checked_before_any_fetch(self, bn):
        """An unknown method fails without touching the source."""
        bn.set_grid(*self.grid)

        class Exploding(geobn.ConstantSource):
            def _fetch(self, grid=None):
                raise AssertionError("source must not be fetched")

        bn.set_input("slope", Exploding(1.0))
        with pytest.raises(ValueError, match="Unknown method"):
            bn.suggest_breakpoints("slope", method="nope")


class TestMosaicSourceEndToEnd:
    """A mosaic behaves like the sources it wraps, including under the auto-grid."""

    @staticmethod
    def _half(value: float, shape: tuple[int, int] = (10, 10)) -> np.ndarray:
        """*value* on the northern half of the array, NaN on the southern half."""
        array = np.full(shape, np.nan, dtype=np.float32)
        array[: shape[0] // 2, :] = value
        return array

    def test_auto_grid_uses_the_mosaic(self, bn):
        """The mosaic seeds the reference grid, as its best source would on its own."""
        coarse_transform = Affine(0.5, 0, 0.0, 0, -0.5, 50.0)
        fine_transform = Affine(0.1, 0, 0.0, 0, -0.1, 50.0)

        bn.set_input(
            "slope",
            geobn.MosaicSource(
                [
                    geobn.ArraySource(
                        np.full((10, 10), 5.0, dtype=np.float32),
                        crs="EPSG:4326",
                        transform=fine_transform,
                    ),
                    geobn.ConstantSource(5.0),
                ]
            ),
        )
        bn.set_input(
            "rainfall",
            geobn.ArraySource(
                np.full((2, 2), 50.0, dtype=np.float32),
                crs="EPSG:4326",
                transform=coarse_transform,
            ),
        )
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        result = bn.infer(query=["fire_risk"])

        assert result.probabilities["fire_risk"].shape[:2] == (10, 10)

    def test_mosaic_is_merged_after_the_grid_is_known(self, bn):
        """The blind probe seeds the grid; the merge still runs on the refetch."""
        transform = Affine(0.1, 0, 0.0, 0, -0.1, 50.0)
        patchy = geobn.ArraySource(self._half(5.0), crs="EPSG:4326", transform=transform)

        bn.set_input("slope", geobn.MosaicSource([patchy, geobn.ConstantSource(15.0)]))
        bn.set_input(
            "rainfall",
            geobn.ArraySource(
                np.full((10, 10), 50.0, dtype=np.float32),
                crs="EPSG:4326",
                transform=transform,
            ),
        )
        bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
        bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])

        result = bn.infer(query=["fire_risk"])

        # The constant covers the southern half, so no pixel is left without data.
        probs = result.probabilities["fire_risk"]
        assert not np.isnan(probs).any()

    def test_fetch_raw_returns_values_and_provenance(self, bn):
        bn.set_grid("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        transform = Affine(0.1, 0, 0.0, 0, -0.1, 50.0)
        mosaic = geobn.MosaicSource(
            [
                geobn.ArraySource(self._half(5.0), crs="EPSG:4326", transform=transform),
                geobn.ConstantSource(15.0),
            ],
            names=["survey", "prior"],
        )

        values, provenance = bn.fetch_raw(mosaic, return_provenance=True)

        assert values.shape == provenance.shape == bn._grid.shape
        assert provenance.dtype == np.int16
        assert set(np.unique(provenance)) == {0, 1}
        assert values[provenance == 0].min() == pytest.approx(5.0)
        assert values[provenance == 1].min() == pytest.approx(15.0)
        assert mosaic.names == ["survey", "prior"]

    def test_fetch_raw_provenance_of_a_single_source(self, bn):
        """A source that is not a mosaic has one layer, so it reports a coverage mask."""
        bn.set_grid("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        transform = Affine(0.1, 0, 0.0, 0, -0.1, 50.0)
        source = geobn.ArraySource(self._half(5.0), crs="EPSG:4326", transform=transform)

        values, provenance = bn.fetch_raw(source, return_provenance=True)

        assert provenance.dtype == np.int16
        np.testing.assert_array_equal(provenance == -1, np.isnan(values))
        np.testing.assert_array_equal(provenance[~np.isnan(values)], 0)

    def test_fetch_raw_without_provenance_returns_a_bare_array(self, bn):
        bn.set_grid("EPSG:4326", 0.1, (0.0, 49.0, 1.0, 50.0))
        mosaic = geobn.MosaicSource([geobn.ConstantSource(42.0)])

        values = bn.fetch_raw(mosaic)

        assert isinstance(values, np.ndarray)
        assert np.all(values == 42.0)
