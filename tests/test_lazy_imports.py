"""geobn imports and runs its network-free parts without pgmpy.

Each check runs in a fresh interpreter with a meta-path finder that makes any
``pgmpy`` import fail, so the modules already imported by the test session
are not affected.
"""
import subprocess
import sys
import textwrap

_BLOCK_PGMPY = textwrap.dedent("""
    import sys

    class _BlockPgmpy:
        def find_spec(self, name, path=None, target=None):
            if name == "pgmpy" or name.startswith("pgmpy."):
                raise ImportError(f"No module named {name!r} (blocked for test)")
            return None

    sys.meta_path.insert(0, _BlockPgmpy())
""")


def _run_without_pgmpy(code: str) -> subprocess.CompletedProcess:
    script = _BLOCK_PGMPY + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
    )


def _assert_ok(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, proc.stderr


def test_import_does_not_load_pgmpy():
    _assert_ok(_run_without_pgmpy("""
        import geobn
        assert "pgmpy" not in sys.modules
    """))


def test_network_free_parts_work_without_pgmpy():
    _assert_ok(_run_without_pgmpy("""
        import numpy as np
        from affine import Affine

        import geobn
        from geobn.discretize import DiscretizationSpec, discretize_array
        from geobn.inference import run_inference_from_table

        values = np.array([[1.0, 5.0], [15.0, np.nan]])
        data = geobn.ArraySource(values).fetch()
        assert data.array.shape == (2, 2)

        bp = geobn.breakpoints.equal_interval(values, 2)
        idx = discretize_array(values, DiscretizationSpec(bp, ["low", "high"]))
        assert idx.tolist() == [[0, 0], [1, -1]]

        table = {"risk": np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)}
        probs = run_inference_from_table(
            table=table,
            node_order=["x"],
            evidence_state_grids={"x": idx},
            nodata_mask=idx < 0,
        )
        np.testing.assert_allclose(probs["risk"][1, 0], [0.2, 0.8])
        assert np.isnan(probs["risk"][1, 1]).all()

        result = geobn.InferenceResult(
            probabilities=probs,
            state_names={"risk": ["low", "high"]},
            crs="EPSG:4326",
            transform=Affine.identity(),
        )
        assert result.mode("risk").shape == (2, 2)
        assert "pgmpy" not in sys.modules
    """))


def test_building_a_network_without_pgmpy_names_it(tmp_path):
    bif = tmp_path / "model.bif"
    bif.write_text("network x {}\\n", encoding="utf-8")
    _assert_ok(_run_without_pgmpy(f"""
        import geobn

        for build in (
            lambda: geobn.GeoBayesianNetwork(object()),
            lambda: geobn.load({str(bif)!r}),
        ):
            try:
                build()
            except ImportError as exc:
                assert "pip install pgmpy" in str(exc), exc
            else:
                raise AssertionError("expected ImportError")
    """))
