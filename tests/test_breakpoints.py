"""Tests for the breakpoint helpers."""
from __future__ import annotations

import numpy as np
import pytest

from geobn.breakpoints import equal_interval, quantile
from geobn.discretize import DiscretizationSpec, discretize_array


class TestQuantile:
    def test_returns_n_plus_one_edges(self):
        assert len(quantile(np.arange(100.0), 4)) == 5

    def test_outer_edges_are_data_min_and_max(self):
        bp = quantile(np.arange(10.0), 3)
        assert bp[0] == 0.0
        assert bp[-1] == 9.0

    def test_bins_hold_roughly_equal_counts(self):
        rng = np.random.default_rng(0)
        arr = rng.gamma(2.0, 8.0, 10_000)  # strongly right-skewed
        spec = DiscretizationSpec(quantile(arr, 4), ["a", "b", "c", "d"])
        counts = np.bincount(discretize_array(arr, spec).ravel(), minlength=4)
        assert counts.min() > 0.9 * 2500
        assert counts.max() < 1.1 * 2500

    def test_nan_is_ignored(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0, np.nan, np.nan])
        assert quantile(arr, 3) == quantile(np.array([0.0, 1.0, 2.0, 3.0]), 3)

    def test_infinite_values_are_ignored(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0, np.inf, -np.inf])
        assert quantile(arr, 3) == quantile(np.array([0.0, 1.0, 2.0, 3.0]), 3)

    def test_shape_does_not_matter(self):
        flat = np.arange(24.0)
        assert quantile(flat.reshape(4, 6), 4) == quantile(flat, 4)

    def test_accepts_a_plain_list(self):
        assert quantile([0, 1, 2, 3, 4, 5, 6, 7], 4) == [0.0, 1.75, 3.5, 5.25, 7.0]

    def test_bounds_replace_the_outer_edges(self):
        bp = quantile(np.arange(10.0), 3, bounds=(-5.0, 90.0))
        assert bp[0] == -5.0
        assert bp[-1] == 90.0
        # Interior edges still come from the data
        assert bp[1:-1] == quantile(np.arange(10.0), 3)[1:-1]

    def test_one_sided_bounds_keep_the_other_side_from_the_data(self):
        bp = quantile(np.arange(10.0), 3, bounds=(0.0, None))
        assert bp[0] == 0.0
        assert bp[-1] == 9.0

    def test_infinite_bounds_are_accepted(self):
        bp = quantile(np.arange(10.0), 3, bounds=(-np.inf, np.inf))
        assert bp[0] == -np.inf
        assert bp[-1] == np.inf

    def test_single_bin(self):
        assert quantile(np.arange(10.0), 1) == [0.0, 9.0]


class TestEqualInterval:
    def test_edges_are_evenly_spaced(self):
        bp = equal_interval(np.array([0.0, 100.0]), 4)
        assert bp == [0.0, 25.0, 50.0, 75.0, 100.0]

    def test_outer_edges_are_data_min_and_max(self):
        bp = equal_interval(np.array([3.0, 5.0, 11.0]), 2)
        assert bp == [3.0, 7.0, 11.0]

    def test_nan_is_ignored(self):
        arr = np.array([0.0, 50.0, np.nan, 100.0])
        assert equal_interval(arr, 4) == [0.0, 25.0, 50.0, 75.0, 100.0]

    def test_bounds_set_the_span(self):
        bp = equal_interval(np.array([10.0, 20.0]), 3, bounds=(0.0, 90.0))
        assert bp == [0.0, 30.0, 60.0, 90.0]

    def test_infinite_bounds_raise(self):
        with pytest.raises(ValueError, match="finite bounds"):
            equal_interval(np.arange(10.0), 3, bounds=(-np.inf, np.inf))

    def test_accepts_a_plain_list(self):
        assert equal_interval([0, 1, 2, 3, 4, 5, 6, 7], 4) == [0.0, 1.75, 3.5, 5.25, 7.0]


class TestValidation:
    @pytest.mark.parametrize("fn", [quantile, equal_interval])
    def test_zero_bins_raises(self, fn):
        with pytest.raises(ValueError, match="at least 1 bin"):
            fn(np.arange(10.0), 0)

    @pytest.mark.parametrize("fn", [quantile, equal_interval])
    def test_non_integer_n_raises(self, fn):
        with pytest.raises(ValueError, match="integer number of bins"):
            fn(np.arange(10.0), 2.5)

    @pytest.mark.parametrize("fn", [quantile, equal_interval])
    def test_all_nan_raises(self, fn):
        with pytest.raises(ValueError, match="No finite values"):
            fn(np.full(10, np.nan), 3)

    @pytest.mark.parametrize("fn", [quantile, equal_interval])
    def test_empty_raises(self, fn):
        with pytest.raises(ValueError, match="No finite values"):
            fn(np.array([]), 3)

    @pytest.mark.parametrize("fn", [quantile, equal_interval])
    def test_reversed_bounds_raise(self, fn):
        with pytest.raises(ValueError, match="below the upper bound"):
            fn(np.arange(10.0), 3, bounds=(90.0, 0.0))

    @pytest.mark.parametrize("fn", [quantile, equal_interval])
    def test_nan_bound_raises(self, fn):
        with pytest.raises(ValueError, match="must not be NaN"):
            fn(np.arange(10.0), 3, bounds=(np.nan, 90.0))

    @pytest.mark.parametrize("fn", [quantile, equal_interval])
    def test_malformed_bounds_raise(self, fn):
        with pytest.raises(ValueError, match=r"\(lo, hi\) tuple"):
            fn(np.arange(10.0), 3, bounds=5.0)

    def test_bounds_narrower_than_the_data_raise(self):
        with pytest.raises(ValueError, match="falls outside the bounds"):
            quantile(np.arange(50.0), 4, bounds=(0.0, 10.0))

    def test_tied_data_raises_naming_the_repeated_value(self):
        arr = np.zeros(1000)
        arr[:300] = np.arange(1.0, 301.0)
        with pytest.raises(ValueError) as excinfo:
            quantile(arr, 4)
        message = str(excinfo.value)
        assert "70% of the values are 0" in message
        assert "quantile bins" in message
        assert "fewer bins" in message

    def test_constant_data_raises(self):
        with pytest.raises(ValueError, match="100% of the values are 7"):
            equal_interval(np.full(100, 7.0), 3)


class TestDiscretizationRoundTrip:
    @pytest.mark.parametrize("fn", [quantile, equal_interval])
    def test_result_is_accepted_as_breakpoints(self, fn):
        rng = np.random.default_rng(1)
        arr = rng.uniform(0.0, 90.0, (32, 32))
        spec = DiscretizationSpec(fn(arr, 3), ["flat", "moderate", "steep"])
        idx = discretize_array(arr, spec)
        assert idx.min() >= 0
        assert idx.max() == 2

    def test_out_of_range_nan_marks_values_beyond_the_data(self):
        # The outer breakpoints are the data's own extremes, so a later value
        # beyond them is out of range.
        spec = DiscretizationSpec(
            quantile(np.arange(10.0), 2), ["low", "high"], out_of_range="nan"
        )
        idx = discretize_array(np.array([-1.0, 5.0, 20.0]), spec)
        np.testing.assert_array_equal(idx, [-1, 1, -1])

    def test_bounds_widen_the_valid_range(self):
        spec = DiscretizationSpec(
            quantile(np.arange(10.0), 2, bounds=(-50.0, 50.0)),
            ["low", "high"],
            out_of_range="nan",
        )
        idx = discretize_array(np.array([-1.0, 5.0, 20.0]), spec)
        np.testing.assert_array_equal(idx, [0, 1, 1])
