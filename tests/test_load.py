"""Tests for geobn.load() — format dispatch, validation and round-trips."""
from __future__ import annotations

import locale
from pathlib import Path

import numpy as np
import pytest
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.readwrite import (
    BIFWriter,
    NETWriter,
    UAIWriter,
    XBNWriter,
    XDSLWriter,
    XMLBIFWriter,
)

import geobn
from geobn.network import _model_hash

# (extension, writer class, write method) for the formats that preserve names.
LOSSLESS_FORMATS = [
    (".bif", BIFWriter, "write_bif"),
    (".xmlbif", XMLBIFWriter, "write_xmlbif"),
    (".net", NETWriter, "write_net"),
    (".xdsl", XDSLWriter, "write_xdsl"),
]


def write_model(model, path, writer_cls, method):
    """Write ``model`` to ``path`` with a pgmpy writer."""
    getattr(writer_cls(model), method)(str(path))
    return path


class TestFormatDispatch:
    """Every supported extension is read with the right pgmpy reader."""

    @pytest.mark.parametrize("suffix,writer_cls,method", LOSSLESS_FORMATS)
    def test_roundtrip_preserves_model(
        self, fire_risk_model, tmp_path, suffix, writer_cls, method
    ):
        path = write_model(fire_risk_model, tmp_path / f"model{suffix}", writer_cls, method)

        bn = geobn.load(path)

        assert isinstance(bn, geobn.GeoBayesianNetwork)
        assert set(bn._model.nodes()) == {"slope", "rainfall", "fire_risk"}
        assert bn._model.get_cpds("fire_risk").state_names["fire_risk"] == [
            "low",
            "medium",
            "high",
        ]
        # Structure, state names and CPD values all survive the round-trip.
        assert _model_hash(bn._model) == _model_hash(fire_risk_model)

    def test_accepts_str_and_path(self, fire_risk_model, tmp_path):
        path = write_model(fire_risk_model, tmp_path / "model.bif", BIFWriter, "write_bif")

        assert set(geobn.load(path)._model.nodes()) == set(
            geobn.load(str(path))._model.nodes()
        )

    def test_suffix_is_case_insensitive(self, fire_risk_model, tmp_path):
        path = write_model(fire_risk_model, tmp_path / "model.BIF", BIFWriter, "write_bif")

        assert set(geobn.load(path)._model.nodes()) == {"slope", "rainfall", "fire_risk"}

    def test_unsupported_extension_raises(self, fire_risk_model, tmp_path):
        path = write_model(fire_risk_model, tmp_path / "model.foo", BIFWriter, "write_bif")

        with pytest.raises(ValueError, match=r"Unsupported .*\.foo"):
            geobn.load(path)

    def test_netica_dne_mentioned_in_error(self, tmp_path):
        path = tmp_path / "model.dne"
        path.write_text("// Netica", encoding="utf-8")

        with pytest.raises(ValueError, match=r"\.dne is not supported"):
            geobn.load(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope.bif"):
            geobn.load(tmp_path / "nope.bif")


class TestXmlSniffing:
    """A bare .xml file is resolved from its root element."""

    @pytest.mark.parametrize(
        "writer_cls,method",
        [
            (XMLBIFWriter, "write_xmlbif"),  # <BIF>
            (XDSLWriter, "write_xdsl"),  # <smile>
            (XBNWriter, "write_xbn"),  # <ANALYSISNOTEBOOK>
        ],
    )
    def test_xml_root_element_selects_reader(
        self, fire_risk_model, tmp_path, writer_cls, method
    ):
        path = write_model(fire_risk_model, tmp_path / "model.xml", writer_cls, method)

        bn = geobn.load(path)

        assert set(bn._model.nodes()) == {"slope", "rainfall", "fire_risk"}

    def test_unknown_root_element_raises(self, tmp_path):
        path = tmp_path / "model.xml"
        path.write_text("<nonsense><a/></nonsense>", encoding="utf-8")

        with pytest.raises(ValueError, match="<NONSENSE>"):
            geobn.load(path)

    def test_not_xml_at_all_raises(self, tmp_path):
        path = tmp_path / "model.xml"
        path.write_text("this is not xml", encoding="utf-8")

        with pytest.raises(ValueError, match="not well-formed XML"):
            geobn.load(path)

    def test_truncated_xml_raises_naming_the_file(self, tmp_path):
        # The root element parses, so sniffing succeeds and the reader fails.
        path = tmp_path / "model.xml"
        path.write_text("<BIF><NETWORK>", encoding="utf-8")

        with pytest.raises(ValueError, match="model.xml"):
            geobn.load(path)


class TestUai:
    """UAI files carry no names; loading one warns about the positional ones."""

    def test_load_warns_about_positional_names(self, fire_risk_model, tmp_path):
        path = write_model(fire_risk_model, tmp_path / "model.uai", UAIWriter, "write_uai")

        with pytest.warns(UserWarning, match="positional names"):
            bn = geobn.load(path)

        assert set(bn._model.nodes()) == {"var_0", "var_1", "var_2"}

    def test_markov_network_raises(self, tmp_path):
        # A minimal MARKOV-type UAI file: 2 binary variables, one pairwise factor.
        path = tmp_path / "markov.uai"
        path.write_text(
            "MARKOV\n2\n2 2\n1\n2 0 1\n\n4\n 0.25 0.25 0.25 0.25\n", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="directed discrete Bayesian network"):
            geobn.load(path)


class TestValidation:
    """Unreadable or invalid files fail at load() with a message naming the file."""

    def test_truncated_file_raises_value_error(self, fire_risk_model, tmp_path):
        path = write_model(fire_risk_model, tmp_path / "model.bif", BIFWriter, "write_bif")
        path.write_text(path.read_text(encoding="utf-8")[:120], encoding="utf-8")

        with pytest.raises(ValueError, match="model.bif"):
            geobn.load(path)

    def test_invalid_cpds_raise_value_error(self, tmp_path):
        # CPD columns do not sum to 1, so check_model() fails.
        path = tmp_path / "bad.bif"
        path.write_text(
            "network unknown {\n}\n"
            "variable a {\n  type discrete [ 2 ] { yes, no };\n}\n"
            "probability ( a ) {\n  table 0.3, 0.3;\n}\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="bad.bif"):
            geobn.load(path)


class TestEncoding:
    """Text formats are decoded as UTF-8, whatever the platform default is."""

    @pytest.mark.parametrize(
        "suffix,writer_cls,method",
        [
            (".bif", BIFWriter, "write_bif"),
            (".net", NETWriter, "write_net"),
        ],
    )
    def test_utf8_comment_loads(
        self, fire_risk_model, tmp_path, suffix, writer_cls, method
    ):
        # pgmpy's text readers open files with the platform default encoding, so a
        # UTF-8 file with non-ASCII characters failed on Windows (cp1252).
        path = write_model(fire_risk_model, tmp_path / f"model{suffix}", writer_cls, method)
        path.write_text(
            "// slope_angle ──► fire_risk (°, 30 % ≥ high)\n"
            + path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        bn = geobn.load(path)

        assert set(bn._model.nodes()) == {"slope", "rainfall", "fire_risk"}

    def test_falls_back_to_platform_encoding(self, fire_risk_model, tmp_path):
        # A legacy file in the platform's own codec, invalid as UTF-8, must still load.
        path = write_model(fire_risk_model, tmp_path / "model.bif", BIFWriter, "write_bif")
        body = path.read_text(encoding="utf-8")
        try:
            raw = ("// café\n" + body).encode(locale.getpreferredencoding(False))
            raw.decode("utf-8")
        except (UnicodeEncodeError, LookupError):
            pytest.skip("platform codec cannot encode the test character")
        except UnicodeDecodeError:
            pass  # what we want: valid in the platform codec, invalid as UTF-8
        else:
            pytest.skip("platform default encoding is UTF-8 compatible")

        path.write_bytes(raw)

        assert set(geobn.load(path)._model.nodes()) == {"slope", "rainfall", "fire_risk"}

    def test_example_model_loads(self):
        """The shipped Lyngen model is UTF-8; it must load on every platform."""
        path = Path(__file__).resolve().parents[1] / "examples/lyngen_alps/avalanche_risk.bif"

        bn = geobn.load(path)

        assert len(bn._model.nodes()) == 9


class TestInferenceAfterLoad:
    """A model loaded from a non-BIF format infers identically to the original."""

    def test_net_roundtrip_gives_same_posteriors(
        self, fire_risk_model, tmp_path, slope_array, rainfall_array, reference_transform
    ):
        path = write_model(fire_risk_model, tmp_path / "model.net", NETWriter, "write_net")

        def source(array):
            return geobn.ArraySource(array, crs="EPSG:4326", transform=reference_transform)

        def configure(bn):
            bn.set_input("slope", source(slope_array))
            bn.set_input("rainfall", source(rainfall_array))
            bn.set_discretization("slope", [0, 10, 30, 90], ["flat", "moderate", "steep"])
            bn.set_discretization("rainfall", [0, 25, 75, 200], ["low", "medium", "high"])
            return bn.infer(query=["fire_risk"])

        expected = configure(geobn.GeoBayesianNetwork(fire_risk_model))
        actual = configure(geobn.load(path))

        np.testing.assert_allclose(
            actual.probabilities["fire_risk"], expected.probabilities["fire_risk"]
        )


class TestDirectModel:
    """GeoBayesianNetwork accepts a pgmpy model directly (documented escape hatch)."""

    def test_wraps_pgmpy_model(self, fire_risk_model):
        bn = geobn.GeoBayesianNetwork(fire_risk_model)

        assert isinstance(bn._model, DiscreteBayesianNetwork)
        assert set(bn._model.nodes()) == {"slope", "rainfall", "fire_risk"}
