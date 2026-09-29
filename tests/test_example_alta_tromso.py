"""Tests for the helper modules of the Alta → Tromsø example.

Route planning and the wave estimate live in ``examples/alta_tromso``, not in
the library; these tests keep them working.  None of them use the network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import geobn

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "alta_tromso"
sys.path.insert(0, str(EXAMPLE))

from met_archive import _parse_ascii, _parse_das  # noqa: E402
from routing import plan_route, snap_to_water  # noqa: E402
from waves import effective_fetch, mean_direction, sheltered_hs, spread_offshore  # noqa: E402


@pytest.fixture
def wall_with_gap() -> np.ndarray:
    """9×9 water grid split by a land wall in column 4, open only at row 7."""
    cost = np.ones((9, 9))
    cost[:, 4] = np.nan
    cost[7, 4] = 1.0
    return cost


class TestPlanRoute:
    def test_path_uses_only_water(self, wall_with_gap):
        path, length, _ = plan_route(wall_with_gap, (1, 1), (1, 7), pixel_size=100.0)
        assert np.isfinite(wall_with_gap[path[:, 0], path[:, 1]]).all()
        assert tuple(path[0]) == (1, 1) and tuple(path[-1]) == (1, 7)
        assert (7, 4) in {tuple(p) for p in path}
        assert length > 600.0   # longer than the straight 600 m through the wall

    def test_steps_are_between_neighbours(self, wall_with_gap):
        path, _, _ = plan_route(wall_with_gap, (0, 0), (8, 8), pixel_size=1.0)
        assert (np.abs(np.diff(path, axis=0)) <= 1).all()

    def test_risk_makes_the_route_detour(self):
        risk = np.zeros((7, 11))
        risk[1:6, 5] = 1.0   # a risky band across the middle, open at rows 0 and 6
        start, goal = (3, 0), (3, 10)
        short, short_len, _ = plan_route(np.ones_like(risk), start, goal, pixel_size=1.0)
        safe, safe_len, _ = plan_route(1 + 8 * risk, start, goal, pixel_size=1.0)
        assert risk[short[:, 0], short[:, 1]].max() == 1.0
        assert risk[safe[:, 0], safe[:, 1]].max() == 0.0
        assert safe_len > short_len

    def test_no_corner_cutting_between_land_pixels(self):
        cost = np.ones((2, 2))
        cost[0, 1] = cost[1, 0] = np.nan
        with pytest.raises(ValueError, match="cannot be reached"):
            plan_route(cost, (0, 0), (1, 1), pixel_size=1.0)

    def test_start_on_land_raises(self, wall_with_gap):
        with pytest.raises(ValueError, match="start"):
            plan_route(wall_with_gap, (0, 4), (1, 7), pixel_size=1.0)

    def test_snap_to_water(self, wall_with_gap):
        water = np.isfinite(wall_with_gap)
        assert snap_to_water(water, (2, 2)) == (2, 2)
        assert snap_to_water(water, (2, 4)) in {(2, 3), (2, 5)}


class TestWaves:
    def test_lee_of_land_has_shorter_fetch(self):
        water = np.ones((40, 40), dtype=bool)
        water[20:24, 10:30] = False   # an island
        fetch = effective_fetch(water, wind_from_deg=0.0, pixel_size=100.0, max_fetch=3000.0)
        assert np.isnan(fetch[21, 20])
        # Just south of the island (in its lee for a north wind) against the
        # same column north of it, which is open to the grid edge.
        assert fetch[25, 20] < fetch[18, 20]

    def test_sheltered_water_gets_smaller_waves(self):
        fetch = np.array([2_000.0, 60_000.0])
        hs = sheltered_hs(np.array([6.0, 6.0]), fetch, wind_speed=10.0)
        assert hs[0] < hs[1]
        assert hs[1] == pytest.approx(6.0)

    def test_spread_offshore_fills_gaps(self):
        hs = np.full((10, 10), np.nan)
        hs[:, :3] = 4.0
        filled = spread_offshore(hs, sigma_px=1.0)
        assert np.isfinite(filled).all()
        assert filled == pytest.approx(4.0)

    def test_mean_direction_wraps_north(self):
        d = mean_direction(np.array([350.0, 10.0]))
        assert min(d, 360.0 - d) == pytest.approx(0.0, abs=1e-6)


class TestOpendapParsing:
    def test_parse_grid_ascii(self):
        text = (
            "Dataset {\n    Grid {\n    } hs;\n} f.nc;\n"
            "---------------------------------------------\n"
            "hs.hs[1][2][3]\n"
            "[0][0], 3.2, 3.1, 3.0\n"
            "[0][1], 2.9, -999.0, 2.7\n\n"
            "hs.time[1]\n1.7E9\n"
        )
        values = _parse_ascii(text, (2, 3))
        np.testing.assert_allclose(values, [[3.2, 3.1, 3.0], [2.9, -999.0, 2.7]])

    def test_parse_das(self):
        text = (
            "Attributes {\n"
            "    hs {\n"
            "        Float32 _FillValue -999.0;\n"
            '        String units "m";\n'
            "        Float32 scale_factor 0.001;\n"
            "    }\n"
            "}\n"
        )
        das = _parse_das(text)
        assert das["hs"] == {"_FillValue": -999.0, "units": "m", "scale_factor": 0.001}


def test_network_loads():
    bn = geobn.load(EXAMPLE / "usv_passage.bif")
    assert bn._model.check_model()


class TestWebMapExport:
    def test_exported_table_matches_the_network(self):
        import run_example
        from webmap import TABLE_NODES, risk_table

        bp = run_example.BREAKPOINTS
        bn = geobn.load(EXAMPLE / "usv_passage.bif")
        for node in TABLE_NODES:
            bn.set_input(node, geobn.ConstantSource(0.0))
            bn.set_discretization(node, bp[node])
        bn.precompute(query=["usv_risk"])
        risk, high = risk_table(bn, bp, [0.0, 0.5, 1.0])

        sizes = [len(bp[n]) - 1 for n in TABLE_NODES]
        assert risk.shape == high.shape == (int(np.prod(sizes)),)
        rng = np.random.default_rng(0)
        for _ in range(25):
            s = [int(rng.integers(k)) for k in sizes]
            # A value inside each state's bin, away from the middle the export uses.
            evidence = {n: bp[n][i] + 0.2 * (bp[n][i + 1] - bp[n][i])
                        for n, i in zip(TABLE_NODES, s)}
            p = bn.query_point(evidence)["usv_risk"]
            k = int(np.ravel_multi_index(s, sizes))
            assert risk[k] == pytest.approx(0.5 * p["medium"] + p["high"], abs=1e-6)
            assert high[k] == pytest.approx(p["high"], abs=1e-6)

    def test_states_put_breakpoints_in_the_upper_state(self):
        from webmap import states

        np.testing.assert_array_equal(
            states(np.array([-1.0, 0.0, 0.99, 1.0, 2.5, 99.0]), [0, 1.0, 2.5, 4.5, 30]),
            [0, 0, 0, 1, 2, 3],
        )

    def test_fetch_bytes_keep_the_exposure(self):
        from webmap import FETCH_MAX, dequantise_fetch, quantise_fetch

        fetch = np.array([0.0, 250.0, 1_000.0, 12_345.0, 50_000.0, FETCH_MAX, np.nan])
        back = dequantise_fetch(quantise_fetch(fetch))
        exposure = lambda f: np.sqrt(np.clip(f / 50_000, 0, 1))  # noqa: E731
        ok = np.isfinite(fetch)
        assert np.abs(exposure(back[ok]) - exposure(fetch[ok])).max() < 0.003
        assert back[-1] == 0.0   # NaN (land) is stored as no fetch
