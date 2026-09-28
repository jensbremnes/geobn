"""A source computed from other sources."""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from .._types import RasterData
from ..grid import GridSpec, align_to_grid
from ._base import DataSource

_log = logging.getLogger(__name__)


class DerivedSource(DataSource):
    """Compute a layer from one or more other sources.

    Each input source is aligned to the reference grid, and *fn* is called
    with the aligned arrays, one per source and in the same order.  It returns
    one array of the same shape, which becomes this source's data::

        depth = geobn.DerivedSource(
            lambda z: np.where(z <= 0, -z, np.nan),   # elevation → depth, land → NaN
            geobn.WCSSource(url, layer="emodnet:mean"),
        )
        bn.set_input("water_depth", depth)

    Nothing is fetched until the source is: the inputs are fetched and *fn*
    runs when :meth:`~geobn.GeoBayesianNetwork.infer` or
    :meth:`~geobn.GeoBayesianNetwork.fetch_raw` needs the data, so a derived
    input can be frozen with :meth:`~geobn.GeoBayesianNetwork.freeze` like
    any other.  Derived sources can be nested, and :mod:`geobn.terrain`
    builds on them for slope, aspect, roughness and TPI.

    The source keeps no disk cache of its own.  Give the input sources a
    ``cache_dir`` to avoid downloading them again; *fn* runs on every fetch.

    Parameters
    ----------
    fn:
        Function taking one ``(H, W)`` float32 array per source and
        returning an ``(H, W)`` array.  NaN marks pixels without data, both in
        the arrays passed in and in the one returned.
    *sources:
        The input sources.  Each is resampled onto the grid with its own
        ``resampling`` method before *fn* sees it.
    valid_range:
        Optional ``(lo, hi)`` tuple applied to the result of *fn*.  Either
        bound may be ``None``.
    """

    probe_only = True

    def __init__(
        self,
        fn: Callable[..., np.ndarray],
        *sources: DataSource,
        valid_range: tuple[float | None, float | None] | None = None,
    ) -> None:
        super().__init__(valid_range=valid_range)
        if not callable(fn):
            raise ValueError(
                f"fn must be a function of the aligned input arrays; got {fn!r}"
            )
        if not sources:
            raise ValueError("DerivedSource needs at least one input source")
        for source in sources:
            if not isinstance(source, DataSource):
                raise ValueError(
                    f"every input of DerivedSource must be a data source; got {source!r}"
                )
        self._fn = fn
        self._sources = list(sources)
        # A blind fetch is possible as long as one input can be fetched
        # without a bbox; that input seeds the automatic grid.
        self.requires_grid = all(source.requires_grid for source in self._sources)

    def _fetch(self, grid: GridSpec | None = None) -> RasterData:
        if grid is None:
            return self._probe()
        arrays = [
            align_to_grid(source.fetch(grid=grid), grid, source.resampling)
            for source in self._sources
        ]
        _log.info("Deriving %s from %d source(s)", _fn_name(self._fn), len(arrays))
        out = np.asarray(self._call(arrays, grid), dtype=np.float32)
        if out.shape != grid.shape:
            raise ValueError(
                f"{_fn_name(self._fn)} returned an array of shape {out.shape}, "
                f"but the grid shape is {grid.shape}.  A DerivedSource function "
                "must return one value per grid pixel."
            )
        return RasterData(array=out, crs=grid.crs, transform=grid.transform)

    def _call(self, arrays: list[np.ndarray], grid: GridSpec) -> np.ndarray:
        """Apply the function.  Overridden by operations that need the grid."""
        return self._fn(*arrays)

    def _probe(self) -> RasterData:
        """Return the first input that can be fetched without a grid.

        This seeds the automatic grid, which needs a CRS and a pixel size
        before the inputs can be aligned and combined.
        """
        for source in self._sources:
            if not source.requires_grid:
                return source.fetch(grid=None)
        raise ValueError(
            "DerivedSource requires a grid context, because all of its inputs "
            "do.  This is provided automatically by GeoBayesianNetwork.infer() "
            "and by fetch_raw()."
        )


class _GridDerivedSource(DerivedSource):
    """A DerivedSource whose function also receives the grid as ``grid=``.

    Used by :mod:`geobn.terrain`, whose operations need the pixel spacing.
    """

    def _call(self, arrays: list[np.ndarray], grid: GridSpec) -> np.ndarray:
        return self._fn(*arrays, grid=grid)


def _fn_name(fn: Callable) -> str:
    name = getattr(fn, "__qualname__", None) or type(fn).__name__
    return f"'{name}'"
