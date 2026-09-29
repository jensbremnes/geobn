"""Read MET Norway model archives on thredds.met.no as geobn sources.

MET Norway publishes its wave, ocean and weather model archives as NetCDF
files behind OPeNDAP.  The model grids are projected (rotated pole for the
wave model, Lambert conformal for MEPS, polar stereographic for NorKyst), so
THREDDS cannot serve them as GeoTIFF over WCS.  ``MetArchiveSource`` asks
OPeNDAP for the block of the grid that covers the study area, reads the
plain-text (``.ascii``) response with numpy and returns it on the model's own
projection.  geobn then reprojects it onto the study grid like any other
raster, so the example needs no NetCDF library.

The source follows the pattern in ``docs/api/sources/index.md``: it
implements ``_fetch`` and ``_cache_key``, so ``cache_dir`` works as it does
for the built-in sources and a second run makes no requests.

Data: MET Norway, licensed under CC BY 4.0.
https://www.met.no/en/free-meteorological-data/Licensing-and-crediting
"""
from __future__ import annotations

import re
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import numpy as np
import requests
from affine import Affine
from pyproj import CRS, Transformer

from geobn._types import RasterData
from geobn.grid import GridSpec
from geobn.sources import DataSource

OPENDAP_ROOT = "https://thredds.met.no/thredds/dodsC/"

# MET Norway asks API users to identify themselves.  The same applies to
# THREDDS: keep requests few and cache what you fetch.
USER_AGENT = "geobn-example/alta_tromso (github.com/jensbremnes/geobn)"

# Seconds between requests to thredds.met.no.
_REQUEST_DELAY = 0.2


def _get(url: str, timeout: float) -> str:
    """GET *url* from THREDDS and return the body as text."""
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    _time.sleep(_REQUEST_DELAY)
    if not response.ok:
        raise RuntimeError(
            f"THREDDS request failed with HTTP {response.status_code}: {url}"
        )
    return response.text


def _constraint(expr: str) -> str:
    """Percent-encode an OPeNDAP constraint so Tomcat accepts the brackets."""
    return quote(expr, safe=",:=")


# Metadata of the datasets opened so far.  Several fields and hours are read
# from the same file, and its coordinates only need fetching once.
_OPEN: dict[str, "_Dataset"] = {}


class _Dataset:
    """Metadata of one OPeNDAP dataset: dimensions, attributes, coordinates."""

    def __init__(self, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout
        self._dds = _get(url + ".dds", timeout)
        self._das = _parse_das(_get(url + ".das", timeout))
        self._coords: dict[str, np.ndarray] = {}

    @classmethod
    def open(cls, url: str, timeout: float) -> "_Dataset":
        """Return the metadata of *url*, fetched once per process."""
        if url not in _OPEN:
            _OPEN[url] = cls(url, timeout)
        return _OPEN[url]

    def dims(self, variable: str) -> list[tuple[str, int]]:
        """Return ``[(dim_name, size), ...]`` for *variable*."""
        match = re.search(
            rf"\b\w+\s+{re.escape(variable)}((?:\[\w+ = \d+\])+);", self._dds
        )
        if match is None:
            raise KeyError(f"{variable!r} is not a variable of {self.url}")
        return [(d, int(n)) for d, n in re.findall(r"\[(\w+) = (\d+)\]", match.group(1))]

    def attrs(self, variable: str) -> dict:
        return self._das.get(variable, {})

    def read(self, variable: str, slices: list[tuple[int, int]]) -> np.ndarray:
        """Read ``variable[s0:e0][s1:e1]...`` (inclusive ends) as float64."""
        expr = variable + "".join(f"[{s}:{e}]" for s, e in slices)
        text = _get(f"{self.url}.ascii?{_constraint(expr)}", self.timeout)
        shape = tuple(e - s + 1 for s, e in slices)
        return _parse_ascii(text, shape)

    def coordinate(self, name: str) -> np.ndarray:
        if name not in self._coords:
            size = dict(self.dims(name))[name]
            self._coords[name] = self.read(name, [(0, size - 1)])
        return self._coords[name]

    def crs(self, variable: str) -> CRS:
        """The CRS of *variable*, from its CF ``grid_mapping`` attributes."""
        mapping = self.attrs(variable).get("grid_mapping")
        if mapping is None:
            return CRS.from_epsg(4326)
        attrs = self.attrs(mapping)
        if "proj4" in attrs:
            return CRS.from_proj4(attrs["proj4"].replace("+type=crs", ""))
        return CRS.from_cf(attrs)

    def times(self) -> list[datetime]:
        values = self.coordinate("time")
        units = self.attrs("time").get("units", "seconds since 1970-01-01")
        step, _, origin = units.partition(" since ")
        seconds = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}[step.strip()]
        epoch = datetime.fromisoformat(
            origin.strip().replace(" ", "T").rstrip("Z").split("+")[0][:19]
        ).replace(tzinfo=timezone.utc)
        return [
            datetime.fromtimestamp(epoch.timestamp() + v * seconds, tz=timezone.utc)
            for v in values
        ]


def _parse_das(text: str) -> dict[str, dict]:
    """Parse an OPeNDAP DAS into ``{variable: {attribute: value}}``."""
    result: dict[str, dict] = {}
    current: dict | None = None
    for raw in text.splitlines():
        line = raw.strip()
        block = re.match(r"^(\w+) \{$", line)
        if block and block.group(1) != "Attributes":
            current = result.setdefault(block.group(1), {})
            continue
        if line == "}":
            current = None
            continue
        attr = re.match(r'^(\w+) (\w+) (.*);$', line)
        if attr and current is not None:
            kind, name, value = attr.groups()
            if kind == "String":
                current[name] = value.strip('"')
            else:
                numbers = [float(v) for v in value.split(",")]
                current[name] = numbers[0] if len(numbers) == 1 else numbers
    return result


def _parse_ascii(text: str, shape: tuple[int, ...]) -> np.ndarray:
    """Parse the values of an OPeNDAP ``.ascii`` response into *shape*.

    The response starts with a header line and a dashed rule, then gives the
    values row by row, each row prefixed by its index (``[0][3], 1.2, ...``).
    A Grid response repeats its map vectors afterwards; only the first
    ``prod(shape)`` values belong to the array.
    """
    body = re.split(r"^-{10,}$", text, maxsplit=1, flags=re.MULTILINE)[-1]
    values: list[float] = []
    count = int(np.prod(shape))
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("Dataset", "}")):
            continue
        if line[0] == "[":
            line = line.split(",", 1)[1] if "," in line else ""
        elif re.match(r"^[A-Za-z_][\w.]*(\[\d+\])*$", line):
            continue   # a variable-name header such as "hs.hs[1][40][60]"
        values.extend(float(v) for v in line.split(",") if v.strip())
        if len(values) >= count:
            break
    if len(values) < count:
        raise RuntimeError(f"OPeNDAP response had {len(values)} values, expected {count}")
    return np.asarray(values[:count], dtype=np.float64).reshape(shape)


class MetArchiveSource(DataSource):
    """One field at one time from a MET Norway model archive on thredds.met.no.

    Parameters
    ----------
    dataset:
        Dataset path below ``/thredds/dodsC/``, e.g.
        ``"ww3_4km_archive_files/2024/11/23/ww3_4km_20241123T00Z.nc"``.
    variables:
        One variable name, or several that *combine* turns into one field
        (e.g. the two current components and ``np.hypot``).
    time:
        The model time to read, in UTC.  It must be one of the file's steps.
    combine:
        Function of the variables' arrays; required with several variables.
    fixed_index:
        Index to take along dimensions other than time, y and x (depth,
        height, ensemble member).  Every such dimension defaults to 0.
    name:
        Identifies the field in the cache key and in messages.
    cache_dir:
        Directory for the disk cache.
    margin:
        Model cells added around the study area, so bilinear resampling has
        neighbours at the edge.
    """

    requires_grid = True

    def __init__(
        self,
        dataset: str,
        variables: str | tuple[str, ...],
        time: datetime,
        combine: Callable[..., np.ndarray] | None = None,
        fixed_index: dict[str, int] | None = None,
        name: str | None = None,
        cache_dir: str | Path | None = None,
        margin: int = 2,
        timeout: float = 120,
        resampling: str = "bilinear",
    ) -> None:
        super().__init__(cache_dir=cache_dir, resampling=resampling)
        self._dataset = dataset
        self._variables = (variables,) if isinstance(variables, str) else tuple(variables)
        if len(self._variables) > 1 and combine is None:
            raise ValueError("combine is required when reading several variables")
        self._time = time.astimezone(timezone.utc)
        self._combine = combine
        self._fixed_index = dict(fixed_index or {})
        self._name = name or "+".join(self._variables)
        self._margin = margin
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"MetArchiveSource({self._name!r}, {self._time:%Y-%m-%d %H:%MZ})"

    def _cache_key(self, grid: GridSpec | None = None) -> dict | None:
        if grid is None:
            return None
        lon_min, lat_min, lon_max, lat_max = grid.extent_wgs84()
        return {
            "thredds": self._dataset,
            "variables": list(self._variables),
            "name": self._name,
            "time": self._time.isoformat(),
            "fixed_index": self._fixed_index,
            "margin": self._margin,
            "lon_min": round(lon_min, 6), "lat_min": round(lat_min, 6),
            "lon_max": round(lon_max, 6), "lat_max": round(lat_max, 6),
        }

    def _fetch(self, grid: GridSpec | None = None) -> RasterData:
        if grid is None:
            raise ValueError("MetArchiveSource needs the grid to choose the area to read.")

        ds = _Dataset.open(OPENDAP_ROOT + self._dataset, self._timeout)
        first = self._variables[0]
        dims = ds.dims(first)
        y_dim, x_dim = dims[-2][0], dims[-1][0]
        crs = ds.crs(first)
        x = ds.coordinate(x_dim)
        y = ds.coordinate(y_dim)
        # Rotated-pole coordinates are in degrees, and pyproj's rotated-pole
        # CRS works in degrees too.  Projected grids may give km.
        if ds.attrs(x_dim).get("units", "") == "km":
            x, y = x * 1000.0, y * 1000.0

        i0, i1, j0, j1 = self._window(grid, crs, x, y)

        times = ds.times()
        try:
            t_index = times.index(self._time)
        except ValueError:
            raise ValueError(
                f"{self._dataset} has no step at {self._time:%Y-%m-%d %H:%MZ}; "
                f"it covers {times[0]:%Y-%m-%d %H:%MZ} to {times[-1]:%Y-%m-%d %H:%MZ}"
            ) from None

        arrays = []
        for variable in self._variables:
            slices = []
            for dim, _size in ds.dims(variable):
                if dim == "time":
                    slices.append((t_index, t_index))
                elif dim == y_dim:
                    slices.append((j0, j1))
                elif dim == x_dim:
                    slices.append((i0, i1))
                else:
                    k = self._fixed_index.get(dim, 0)
                    slices.append((k, k))
            raw = ds.read(variable, slices).reshape(j1 - j0 + 1, i1 - i0 + 1)
            arrays.append(_unpack(raw, ds.attrs(variable)))

        field = self._combine(*arrays) if self._combine else arrays[0]

        # geobn expects north-up rasters: flip when the y coordinate ascends.
        xs, ys = x[i0:i1 + 1], y[j0:j1 + 1]
        dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
        if dy > 0:
            field, ys, dy = field[::-1], ys[::-1], -dy
        transform = Affine(dx, 0.0, float(xs[0]) - dx / 2, 0.0, dy, float(ys[0]) - dy / 2)
        return RasterData(array=field.astype(np.float32), crs=crs.to_wkt(), transform=transform)

    def _window(self, grid: GridSpec, crs: CRS, x: np.ndarray, y: np.ndarray):
        """Index range of the model cells that cover *grid*, plus the margin."""
        h, w = grid.shape
        t = grid.transform
        edge = np.linspace(0, 1, 50)
        cols = np.concatenate([edge * w, edge * w, np.zeros(50), np.full(50, w)])
        rows = np.concatenate([np.zeros(50), np.full(50, h), edge * h, edge * h])
        gx, gy = t * (cols, rows)
        mx, my = Transformer.from_crs(grid.crs, crs, always_xy=True).transform(gx, gy)

        def span(coord: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
            order = np.argsort(coord)
            a = np.searchsorted(coord[order], lo) - 1 - self._margin
            b = np.searchsorted(coord[order], hi) + self._margin
            a, b = max(a, 0), min(b, coord.size - 1)
            picked = order[a:b + 1]
            return int(picked.min()), int(picked.max())

        i0, i1 = span(x, float(np.min(mx)), float(np.max(mx)))
        j0, j1 = span(y, float(np.min(my)), float(np.max(my)))
        return i0, i1, j0, j1


def _unpack(raw: np.ndarray, attrs: dict) -> np.ndarray:
    """Apply CF ``_FillValue``/``missing_value`` and ``scale_factor``/``add_offset``."""
    out = raw.astype(np.float64)
    for key in ("_FillValue", "missing_value"):
        if key in attrs:
            out[raw == attrs[key]] = np.nan
    out *= attrs.get("scale_factor", 1.0)
    out += attrs.get("add_offset", 0.0)
    return out
