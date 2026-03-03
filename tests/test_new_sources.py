"""Tests for the new data sources added in the Norwegian/ocean data plan."""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from affine import Affine

import geobn
from geobn._types import RasterData
from geobn.grid import GridSpec
from geobn.sources.wcs_source import WCSSource
from geobn.sources.kartverket_source import KartverketDTMSource
from geobn.sources.emodnet_source import EMODnetBathymetrySource, EMODnetShippingDensitySource
from geobn.sources.met_norway_source import (
    METOceanForecastSource,
    METLocationForecastSource,
)
from geobn.sources.barentswatch_source import BarentswatchAISSource
from geobn.sources.copernicus_source import CopernicusMarineSource
from geobn.sources.hubocean_source import HubOceanSource


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid() -> GridSpec:
    """5×5 grid over a small Norwegian coastal area in WGS84."""
    transform = Affine(0.1, 0, 5.0, 0, -0.1, 62.0)
    return GridSpec(crs="EPSG:4326", transform=transform, shape=(5, 5))


def _make_geotiff_bytes(array: np.ndarray) -> bytes:
    """Build minimal in-memory GeoTIFF bytes using rasterio."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    H, W = array.shape
    transform = from_bounds(5.0, 60.0, 6.0, 62.0, W, H)
    buf = io.BytesIO()
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=H,
            width=W,
            count=1,
            dtype=np.float32,
            crs="EPSG:4326",
            transform=transform,
        ) as ds:
            ds.write(array.astype(np.float32), 1)
        buf.write(memfile.read())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# WCSSource
# ---------------------------------------------------------------------------

class TestWCSSource:
    def test_requires_grid(self):
        src = WCSSource(url="http://example.com/wcs", layer="test")
        with pytest.raises(ValueError, match="grid context"):
            src.fetch(grid=None)

    def test_requires_rasterio(self, small_grid):
        import importlib, sys
        if importlib.util.find_spec("rasterio") is not None:
            pytest.skip("rasterio is installed")
        src = WCSSource(url="http://example.com/wcs", layer="test")
        with pytest.raises(ImportError, match="rasterio"):
            src.fetch(grid=small_grid)

    def test_http_error_raises_runtime_error(self, small_grid):
        pytest.importorskip("rasterio")
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        with patch("requests.get", return_value=mock_resp):
            src = WCSSource(url="http://example.com/wcs", layer="test")
            with pytest.raises(RuntimeError, match="HTTP 500"):
                src.fetch(grid=small_grid)

    def test_successful_fetch_returns_raster_data(self, small_grid):
        pytest.importorskip("rasterio")
        array = np.ones((5, 5), dtype=np.float32) * 100.0
        tiff_bytes = _make_geotiff_bytes(array)

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.content = tiff_bytes
        with patch("requests.get", return_value=mock_resp):
            src = WCSSource(url="http://example.com/wcs", layer="test")
            data = src.fetch(grid=small_grid)

        assert isinstance(data, RasterData)
        assert data.array.dtype == np.float32
        assert data.crs is not None
        assert data.transform is not None

    def test_v2_params_contain_subset(self, small_grid):
        pytest.importorskip("rasterio")
        array = np.ones((5, 5), dtype=np.float32)
        tiff_bytes = _make_geotiff_bytes(array)

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.content = tiff_bytes

        captured = {}
        def fake_get(url, params=None, **kwargs):
            captured["params"] = params
            return mock_resp

        with patch("requests.get", side_effect=fake_get):
            src = WCSSource(url="http://example.com/wcs", layer="mycov", version="2.0.1")
            src.fetch(grid=small_grid)

        params = captured["params"]
        assert params["REQUEST"] == "GetCoverage"
        assert params["COVERAGEID"] == "mycov"
        assert "SUBSET" in params

    def test_v1_params_contain_bbox(self, small_grid):
        pytest.importorskip("rasterio")
        array = np.ones((5, 5), dtype=np.float32)
        tiff_bytes = _make_geotiff_bytes(array)

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.content = tiff_bytes

        captured = {}
        def fake_get(url, params=None, **kwargs):
            captured["params"] = params
            return mock_resp

        with patch("requests.get", side_effect=fake_get):
            src = WCSSource(url="http://example.com/wcs", layer="mycov", version="1.1.1")
            src.fetch(grid=small_grid)

        params = captured["params"]
        assert "BBOX" in params
        assert params["CRS"] == "EPSG:4326"


# ---------------------------------------------------------------------------
# KartverketDTMSource
# ---------------------------------------------------------------------------

class TestKartverketDTMSource:
    def test_invalid_layer_raises_value_error(self):
        with pytest.raises(ValueError, match="dtm10"):
            KartverketDTMSource(layer="nonexistent")

    def test_valid_layers_accepted(self):
        for layer in ("dtm10", "dom10"):
            src = KartverketDTMSource(layer=layer)
            assert src._layer == layer

    def test_nodata_sentinel_replaced_with_nan(self, small_grid):
        pytest.importorskip("rasterio")
        # Array with valid values and sentinel nodata values
        array = np.array([[100.0, -9999.0, 50.0, 200.0, -600.0],
                          [30.0, 15.0, 9001.0, 80.0, 120.0],
                          [40.0, 60.0, 70.0, 90.0, 110.0],
                          [20.0, 25.0, 35.0, 45.0, 55.0],
                          [10.0, 5.0, 8.0, 12.0, 18.0]], dtype=np.float32)
        tiff_bytes = _make_geotiff_bytes(array)

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.content = tiff_bytes

        with patch("requests.get", return_value=mock_resp):
            src = KartverketDTMSource()
            data = src.fetch(grid=small_grid)

        result = data.array
        assert np.isnan(result[0, 1])   # -9999 → NaN
        assert np.isnan(result[0, 4])   # -600  → NaN
        assert np.isnan(result[1, 2])   # 9001  → NaN
        assert not np.isnan(result[0, 0])  # 100 is valid
        assert not np.isnan(result[0, 2])  # 50  is valid

    def test_in_geobn_namespace(self):
        assert hasattr(geobn, "KartverketDTMSource")


# ---------------------------------------------------------------------------
# EMODnetBathymetrySource
# ---------------------------------------------------------------------------

class TestEMODnetBathymetrySource:
    def test_invalid_layer_raises_value_error(self):
        with pytest.raises(ValueError, match="emodnet:mean"):
            EMODnetBathymetrySource(layer="emodnet:invalid")

    def test_valid_layers_accepted(self):
        for layer in ("emodnet:mean", "emodnet:stdev"):
            src = EMODnetBathymetrySource(layer=layer)
            assert src._layer == layer

    def test_nodata_sentinel_replaced_with_nan(self, small_grid):
        pytest.importorskip("rasterio")
        array = np.array([[-200.0, -50.0, 9001.0, -100.0, -16000.0],
                          [-300.0, -80.0, -120.0, -250.0, -400.0],
                          [-500.0, -600.0, -700.0, -800.0, -900.0],
                          [-1000.0, -1100.0, -1200.0, -1300.0, -1400.0],
                          [-1500.0, -1600.0, -1700.0, -1800.0, -1900.0]], dtype=np.float32)
        tiff_bytes = _make_geotiff_bytes(array)

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.content = tiff_bytes

        with patch("requests.get", return_value=mock_resp):
            src = EMODnetBathymetrySource()
            data = src.fetch(grid=small_grid)

        result = data.array
        assert np.isnan(result[0, 2])   # 9001  → NaN
        assert np.isnan(result[0, 4])   # -16000 → NaN
        assert not np.isnan(result[0, 0])  # -200 is valid
        assert not np.isnan(result[0, 1])  # -50 is valid

    def test_in_geobn_namespace(self):
        assert hasattr(geobn, "EMODnetBathymetrySource")


# ---------------------------------------------------------------------------
# METOceanForecastSource
# ---------------------------------------------------------------------------

def _ocean_forecast_response(variable: str, value: float) -> dict:
    return {
        "properties": {
            "timeseries": [
                {
                    "time": "2024-01-01T12:00:00Z",
                    "data": {
                        "instant": {
                            "details": {variable: value}
                        }
                    },
                }
            ]
        }
    }


class TestMETOceanForecastSource:
    def test_invalid_variable_raises_value_error(self):
        with pytest.raises(ValueError, match="sea_water_temperature"):
            METOceanForecastSource(variable="invalid_var")

    def test_requires_grid(self):
        src = METOceanForecastSource(variable="sea_water_temperature")
        with pytest.raises(ValueError, match="grid context"):
            src.fetch(grid=None)

    def test_single_point_no_transform(self):
        src = METOceanForecastSource(variable="sea_water_temperature", sample_points=1)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = _ocean_forecast_response("sea_water_temperature", 8.5)

        grid = GridSpec(crs="EPSG:4326", transform=Affine(1.0, 0, 5.0, 0, -1.0, 62.0), shape=(3, 3))
        with patch("requests.get", return_value=mock_resp):
            data = src.fetch(grid=grid)

        assert data.crs is None
        assert data.transform is None
        assert data.array.shape == (1, 1)
        assert data.array[0, 0] == pytest.approx(8.5)

    def test_outside_coverage_returns_nan(self):
        src = METOceanForecastSource(variable="sea_surface_wave_height", sample_points=1)
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.ok = False

        grid = GridSpec(crs="EPSG:4326", transform=Affine(1.0, 0, 5.0, 0, -1.0, 62.0), shape=(3, 3))
        with patch("requests.get", return_value=mock_resp):
            data = src.fetch(grid=grid)

        assert np.isnan(data.array[0, 0])

    def test_multi_point_returns_epsg4326(self):
        src = METOceanForecastSource(variable="sea_water_temperature", sample_points=2)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = _ocean_forecast_response("sea_water_temperature", 10.0)

        grid = GridSpec(crs="EPSG:4326", transform=Affine(0.5, 0, 5.0, 0, -0.5, 62.0), shape=(4, 4))
        with patch("requests.get", return_value=mock_resp), \
             patch("time.sleep"):
            data = src.fetch(grid=grid)

        assert data.crs == "EPSG:4326"
        assert data.array.shape == (2, 2)

    def test_in_geobn_namespace(self):
        assert hasattr(geobn, "METOceanForecastSource")


# ---------------------------------------------------------------------------
# METLocationForecastSource
# ---------------------------------------------------------------------------

def _location_forecast_response(variable: str, value: float, precipitation: bool = False) -> dict:
    if precipitation:
        details_key = "next_1_hours"
        instant_details = {}
        next_1h = {"details": {variable: value}}
    else:
        details_key = "instant"
        instant_details = {variable: value}
        next_1h = {}

    return {
        "properties": {
            "timeseries": [
                {
                    "time": "2024-01-01T12:00:00Z",
                    "data": {
                        "instant": {"details": instant_details},
                        "next_1_hours": next_1h,
                    },
                }
            ]
        }
    }


class TestMETLocationForecastSource:
    def test_invalid_variable_raises_value_error(self):
        with pytest.raises(ValueError, match="air_temperature"):
            METLocationForecastSource(variable="bad_var")

    def test_requires_grid(self):
        src = METLocationForecastSource(variable="air_temperature")
        with pytest.raises(ValueError, match="grid context"):
            src.fetch(grid=None)

    def test_air_temperature_from_instant(self):
        src = METLocationForecastSource(variable="air_temperature", sample_points=1)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _location_forecast_response("air_temperature", 15.3)

        grid = GridSpec(crs="EPSG:4326", transform=Affine(1.0, 0, 5.0, 0, -1.0, 62.0), shape=(3, 3))
        with patch("requests.get", return_value=mock_resp):
            data = src.fetch(grid=grid)

        assert data.array[0, 0] == pytest.approx(15.3)

    def test_precipitation_from_next_1_hours(self):
        src = METLocationForecastSource(variable="precipitation_amount", sample_points=1)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _location_forecast_response(
            "precipitation_amount", 2.5, precipitation=True
        )

        grid = GridSpec(crs="EPSG:4326", transform=Affine(1.0, 0, 5.0, 0, -1.0, 62.0), shape=(3, 3))
        with patch("requests.get", return_value=mock_resp):
            data = src.fetch(grid=grid)

        assert data.array[0, 0] == pytest.approx(2.5)

    def test_in_geobn_namespace(self):
        assert hasattr(geobn, "METLocationForecastSource")


# ---------------------------------------------------------------------------
# BarentswatchAISSource
# ---------------------------------------------------------------------------

class TestBarentswatchAISSource:
    def test_invalid_metric_raises_value_error(self):
        with pytest.raises(ValueError, match="density"):
            BarentswatchAISSource(
                client_id="id", client_secret="secret", metric="invalid"
            )

    def test_requires_grid(self):
        src = BarentswatchAISSource(client_id="id", client_secret="secret")
        with pytest.raises(ValueError, match="grid context"):
            src.fetch(grid=None)

    def _make_mocks(self, vessels: list[dict]):
        token_resp = MagicMock()
        token_resp.ok = True
        token_resp.json.return_value = {"access_token": "tok123"}

        ais_resp = MagicMock()
        ais_resp.ok = True
        ais_resp.json.return_value = vessels

        return token_resp, ais_resp

    def test_count_metric_counts_vessels(self):
        grid = GridSpec(
            crs="EPSG:4326",
            transform=Affine(0.1, 0, 5.0, 0, -0.1, 62.0),
            shape=(5, 5),
        )
        vessels = [
            {"latitude": 61.95, "longitude": 5.05, "speedOverGround": 5.0},
            {"latitude": 61.95, "longitude": 5.15, "speedOverGround": 8.0},
        ]
        token_resp, ais_resp = self._make_mocks(vessels)

        with patch("requests.post", return_value=token_resp), \
             patch("requests.get", return_value=ais_resp):
            src = BarentswatchAISSource(
                client_id="id", client_secret="secret", metric="count"
            )
            data = src.fetch(grid=grid)

        assert data.crs == grid.crs
        assert data.transform == grid.transform
        assert data.array.sum() == pytest.approx(2.0)

    def test_empty_area_returns_zero_array(self):
        grid = GridSpec(
            crs="EPSG:4326",
            transform=Affine(0.1, 0, 5.0, 0, -0.1, 62.0),
            shape=(5, 5),
        )
        token_resp, ais_resp = self._make_mocks([])

        with patch("requests.post", return_value=token_resp), \
             patch("requests.get", return_value=ais_resp):
            src = BarentswatchAISSource(
                client_id="id", client_secret="secret", metric="density"
            )
            data = src.fetch(grid=grid)

        assert np.all(data.array == 0.0)

    def test_speed_metric_computes_mean(self):
        grid = GridSpec(
            crs="EPSG:4326",
            transform=Affine(0.1, 0, 5.0, 0, -0.1, 62.0),
            shape=(5, 5),
        )
        # Both vessels in same pixel (row=0, col=0 for lat≈61.95, lon≈5.05)
        vessels = [
            {"latitude": 61.95, "longitude": 5.05, "speedOverGround": 4.0},
            {"latitude": 61.95, "longitude": 5.05, "speedOverGround": 6.0},
        ]
        token_resp, ais_resp = self._make_mocks(vessels)

        with patch("requests.post", return_value=token_resp), \
             patch("requests.get", return_value=ais_resp):
            src = BarentswatchAISSource(
                client_id="id", client_secret="secret", metric="speed"
            )
            data = src.fetch(grid=grid)

        # Pixel with vessels should have mean speed = 5.0
        filled = data.array[~np.isnan(data.array)]
        assert len(filled) == 1
        assert filled[0] == pytest.approx(5.0)

    def test_vessel_type_filter(self):
        grid = GridSpec(
            crs="EPSG:4326",
            transform=Affine(0.1, 0, 5.0, 0, -0.1, 62.0),
            shape=(5, 5),
        )
        vessels = [
            {"latitude": 61.95, "longitude": 5.05, "speedOverGround": 5.0, "shipType": 70},
            {"latitude": 61.85, "longitude": 5.15, "speedOverGround": 3.0, "shipType": 30},
        ]
        token_resp, ais_resp = self._make_mocks(vessels)

        with patch("requests.post", return_value=token_resp), \
             patch("requests.get", return_value=ais_resp):
            src = BarentswatchAISSource(
                client_id="id", client_secret="secret",
                metric="count", vessel_types=[70],
            )
            data = src.fetch(grid=grid)

        assert data.array.sum() == pytest.approx(1.0)

    def test_in_geobn_namespace(self):
        assert hasattr(geobn, "BarentswatchAISSource")


# ---------------------------------------------------------------------------
# CopernicusMarineSource
# ---------------------------------------------------------------------------

class TestCopernicusMarineSource:
    def test_missing_credentials_raises_value_error(self, small_grid):
        src = CopernicusMarineSource(
            dataset_id="some_dataset",
            variable="temperature",
            datetime="2024-01-01T00:00:00",
        )
        # Clear any env vars that might provide credentials
        with patch.dict("os.environ", {}, clear=True):
            src._username = None
            src._password = None
            with pytest.raises(ValueError, match="credentials"):
                src.fetch(grid=small_grid)

    def test_requires_grid(self):
        src = CopernicusMarineSource(
            dataset_id="ds", variable="var", datetime="2024-01-01T00:00:00",
            username="u", password="p",
        )
        with pytest.raises(ValueError, match="grid context"):
            src.fetch(grid=None)

    def test_missing_copernicusmarine_raises_import_error(self, small_grid):
        import sys
        src = CopernicusMarineSource(
            dataset_id="ds", variable="var", datetime="2024-01-01T00:00:00",
            username="u", password="p",
        )
        with patch.dict(sys.modules, {"copernicusmarine": None}):
            with pytest.raises(ImportError, match="copernicusmarine"):
                src.fetch(grid=small_grid)

    def test_successful_fetch(self, small_grid):
        pytest.importorskip("xarray")
        import xarray as xr

        lats = np.array([61.9, 61.8, 61.7, 61.6, 61.5])
        lons = np.array([5.0, 5.1, 5.2, 5.3, 5.4])
        data_values = np.ones((5, 5), dtype=np.float32) * 12.0

        da = xr.DataArray(
            data_values,
            coords={"latitude": lats, "longitude": lons},
            dims=["latitude", "longitude"],
        )
        mock_ds = xr.Dataset({"temperature": da})

        mock_cm = MagicMock()
        mock_cm.open_dataset.return_value = mock_ds

        import sys
        with patch.dict(sys.modules, {"copernicusmarine": mock_cm}):
            src = CopernicusMarineSource(
                dataset_id="ds", variable="temperature",
                datetime="2024-01-01T00:00:00",
                username="u", password="p",
            )
            result = src.fetch(grid=small_grid)

        assert isinstance(result, RasterData)
        assert result.array.shape == (5, 5)
        assert result.crs == "EPSG:4326"

    def test_in_geobn_namespace(self):
        assert hasattr(geobn, "CopernicusMarineSource")


# ---------------------------------------------------------------------------
# HubOceanSource
# ---------------------------------------------------------------------------

class TestHubOceanSource:
    def test_missing_api_key_raises_value_error(self, small_grid):
        src = HubOceanSource(dataset_id="ds", variable="var")
        src._api_key = None
        with pytest.raises(ValueError, match="API key"):
            src.fetch(grid=small_grid)

    def test_requires_grid(self):
        src = HubOceanSource(dataset_id="ds", variable="var", api_key="key")
        with pytest.raises(ValueError, match="grid context"):
            src.fetch(grid=None)

    def test_missing_pystac_client_raises_import_error(self, small_grid):
        import sys
        src = HubOceanSource(dataset_id="ds", variable="var", api_key="key")
        with patch.dict(sys.modules, {"pystac_client": None}):
            with pytest.raises(ImportError, match="pystac"):
                src.fetch(grid=small_grid)

    def test_no_stac_items_raises_value_error(self, small_grid):
        import sys

        mock_pystac = MagicMock()
        mock_catalog = MagicMock()
        mock_search = MagicMock()
        mock_search.items.return_value = []
        mock_catalog.search.return_value = mock_search
        mock_pystac.Client.open.return_value = mock_catalog

        with patch.dict(sys.modules, {"pystac_client": mock_pystac}):
            src = HubOceanSource(dataset_id="ds", variable="var", api_key="key")
            with pytest.raises(ValueError, match="No STAC items"):
                src.fetch(grid=small_grid)

    def test_missing_variable_raises_value_error(self, small_grid):
        import sys
        import xarray as xr

        lats = np.array([61.9, 61.8, 61.7, 61.6, 61.5])
        lons = np.array([5.0, 5.1, 5.2, 5.3, 5.4])
        da = xr.DataArray(
            np.ones((5, 5), dtype=np.float32),
            coords={"latitude": lats, "longitude": lons},
            dims=["latitude", "longitude"],
        )
        mock_ds = xr.Dataset({"other_var": da})

        mock_asset = MagicMock()
        mock_asset.href = "http://example.com/data.nc"
        mock_item = MagicMock()
        mock_item.assets = {"data": mock_asset}

        mock_pystac = MagicMock()
        mock_catalog = MagicMock()
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]
        mock_catalog.search.return_value = mock_search
        mock_pystac.Client.open.return_value = mock_catalog

        with patch.dict(sys.modules, {"pystac_client": mock_pystac}), \
             patch("xarray.open_dataset", return_value=mock_ds):
            src = HubOceanSource(dataset_id="ds", variable="missing_var", api_key="key")
            with pytest.raises(ValueError, match="missing_var"):
                src.fetch(grid=small_grid)

    def test_successful_fetch(self, small_grid):
        import sys
        import xarray as xr

        lats = np.array([61.9, 61.8, 61.7, 61.6, 61.5])
        lons = np.array([5.0, 5.1, 5.2, 5.3, 5.4])
        da = xr.DataArray(
            np.ones((5, 5), dtype=np.float32) * 7.3,
            coords={"latitude": lats, "longitude": lons},
            dims=["latitude", "longitude"],
        )
        mock_ds = xr.Dataset({"salinity": da})

        mock_asset = MagicMock()
        mock_asset.href = "http://example.com/data.nc"
        mock_item = MagicMock()
        mock_item.assets = {"data": mock_asset}

        mock_pystac = MagicMock()
        mock_catalog = MagicMock()
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]
        mock_catalog.search.return_value = mock_search
        mock_pystac.Client.open.return_value = mock_catalog

        with patch.dict(sys.modules, {"pystac_client": mock_pystac}), \
             patch("xarray.open_dataset", return_value=mock_ds):
            src = HubOceanSource(dataset_id="ds", variable="salinity", api_key="key")
            result = src.fetch(grid=small_grid)

        assert isinstance(result, RasterData)
        assert result.array.shape == (5, 5)
        assert result.crs == "EPSG:4326"

    def test_in_geobn_namespace(self):
        assert hasattr(geobn, "HubOceanSource")


# ---------------------------------------------------------------------------
# EMODnetShippingDensitySource
# ---------------------------------------------------------------------------

class TestEMODnetShippingDensitySource:
    def test_invalid_ship_type_raises_value_error(self):
        with pytest.raises(ValueError, match="ship_type"):
            EMODnetShippingDensitySource(ship_type="submarine")

    def test_valid_ship_types_accepted(self):
        for ship_type in ("all", "cargo", "tanker", "fishing", "passenger", "highspeed"):
            src = EMODnetShippingDensitySource(ship_type=ship_type)
            assert src._ship_type == ship_type

    def test_layer_name_pattern(self):
        src = EMODnetShippingDensitySource(ship_type="cargo", year=2021)
        assert src._wcs._layer == "emodnet:vessel_density_cargo_2021_annual_avg"

    def test_requires_grid(self):
        pytest.importorskip("rasterio")
        src = EMODnetShippingDensitySource()
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 400
        mock_resp.text = "Missing grid"
        with pytest.raises(ValueError, match="grid context"):
            src.fetch(grid=None)

    def test_http_error_raises_runtime_error(self, small_grid):
        pytest.importorskip("rasterio")
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        with patch("requests.get", return_value=mock_resp):
            src = EMODnetShippingDensitySource()
            with pytest.raises(RuntimeError, match="HTTP 500"):
                src.fetch(grid=small_grid)

    def test_nodata_sentinel_replaced_with_nan(self, small_grid):
        pytest.importorskip("rasterio")
        array = np.array([[1.5,  -0.5, 3.0,  0.0,   2.0],
                          [0.8,   1.0, 2.0,  1e7,   0.5],
                          [0.3,   0.1, 0.9,  0.7,   1.2],
                          [2.1,   0.4, 0.6,  0.2,   0.1],
                          [0.05,  0.9, 1.1, -1.0,   0.3]], dtype=np.float32)
        tiff_bytes = _make_geotiff_bytes(array)

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.content = tiff_bytes

        with patch("requests.get", return_value=mock_resp):
            src = EMODnetShippingDensitySource()
            data = src.fetch(grid=small_grid)

        result = data.array
        assert np.isnan(result[0, 1])   # -0.5  → NaN (negative)
        assert np.isnan(result[1, 3])   # 1e7   → NaN (> 1e6)
        assert np.isnan(result[4, 3])   # -1.0  → NaN (negative)
        assert not np.isnan(result[0, 0])  # 1.5  is valid
        assert not np.isnan(result[0, 3])  # 0.0  is valid (zero density)

    def test_successful_fetch_returns_raster_data(self, small_grid):
        pytest.importorskip("rasterio")
        array = np.ones((5, 5), dtype=np.float32) * 2.5
        tiff_bytes = _make_geotiff_bytes(array)

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.content = tiff_bytes
        with patch("requests.get", return_value=mock_resp):
            src = EMODnetShippingDensitySource(ship_type="fishing", year=2022)
            data = src.fetch(grid=small_grid)

        assert isinstance(data, RasterData)
        assert data.array.dtype == np.float32
        assert data.crs is not None
        assert data.transform is not None
        assert np.allclose(data.array, 2.5)

    def test_in_geobn_namespace(self):
        assert hasattr(geobn, "EMODnetShippingDensitySource")


# ---------------------------------------------------------------------------
# Namespace smoke test: all 9 new sources importable from geobn
# ---------------------------------------------------------------------------

def test_all_new_sources_in_geobn_namespace():
    expected = [
        "WCSSource",
        "KartverketDTMSource",
        "EMODnetBathymetrySource",
        "EMODnetShippingDensitySource",
        "METOceanForecastSource",
        "METLocationForecastSource",
        "CopernicusMarineSource",
        "BarentswatchAISSource",
        "HubOceanSource",
    ]
    for name in expected:
        assert hasattr(geobn, name), f"geobn.{name} not found"
