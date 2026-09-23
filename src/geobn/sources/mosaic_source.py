"""Priority mosaic over several data sources."""
from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np

from .._types import RasterData
from ..grid import GridSpec, align_to_grid
from ._base import DataSource, _apply_valid_range

_log = logging.getLogger(__name__)

_ON_ERROR = ("raise", "skip")


class MosaicSource(DataSource):
    """Combine several sources into one layer, best source first.

    Each source is aligned to the reference grid, and the mosaic takes for
    every pixel the value of the first source that has data there.  This
    covers the common case of a node whose data comes from layers of
    decreasing quality and increasing coverage: a local survey where it
    exists, a chart elsewhere, a regional model everywhere.

    Alongside the values, the mosaic produces a provenance layer recording
    which source supplied each pixel.  Read both with
    ``bn.fetch_raw(source, return_provenance=True)``.

    Sources are fetched in order and only while pixels remain uncovered, so a
    lower-priority source is not fetched at all when the ones above it already
    cover the grid.

    Parameters
    ----------
    sources:
        The sources to combine, highest priority first.  A source that covers
        the whole grid, such as :class:`~geobn.ConstantSource`, is the usual
        last entry.
    names:
        Optional labels for the provenance layer, one per source.  They must
        be unique.  Without them the sources are labelled by class and
        position, e.g. ``"RasterSource[0]"``.
    on_error:
        What to do when a source fails to fetch.  ``"raise"`` (default)
        propagates the error, so a mistyped path is not covered up by the
        source below it.  ``"skip"`` treats the source as having no data and
        issues a :class:`UserWarning` naming it, for a layer that is optional
        or sometimes unreachable.
    valid_range:
        Optional ``(lo, hi)`` tuple applied to the combined values, after the
        sources have been merged.  Pixels it masks are NaN in the values and
        ``-1`` in the provenance layer.  A source can carry its own
        ``valid_range`` as well, which applies before the merge and therefore
        lets the next source fill in the pixels it masks.
    """

    probe_only = True

    def __init__(
        self,
        sources: Sequence[DataSource],
        names: Sequence[str] | None = None,
        on_error: str = "raise",
        valid_range: tuple[float | None, float | None] | None = None,
    ) -> None:
        super().__init__(valid_range=valid_range)
        self._sources = _check_sources(sources)
        self._names = _check_names(names, self._sources)
        self._on_error = _check_on_error(on_error)
        # A blind fetch is possible as long as one source can be fetched
        # without a bbox; that source seeds the automatic grid.
        self.requires_grid = all(source.requires_grid for source in self._sources)

    @property
    def names(self) -> list[str]:
        """Labels of the sources in priority order, indexed by the provenance layer."""
        return list(self._names)

    def fetch_with_provenance(
        self, grid: GridSpec | None = None
    ) -> tuple[RasterData, np.ndarray]:
        """Return the combined data together with the provenance layer.

        Parameters
        ----------
        grid:
            Reference grid.  Required: the sources have to be aligned to a
            common grid before they can be compared pixel by pixel.

        Returns
        -------
        tuple[RasterData, np.ndarray]
            The mosaicked data, and an int16 array of the same shape holding
            the index of the source each pixel came from, or ``-1`` where no
            source had data.
        """
        if grid is None:
            raise ValueError(_NEEDS_GRID)
        data, provenance = self._combine(grid)
        data = _apply_valid_range(data, self._valid_range)
        provenance[np.isnan(data.array)] = -1
        return data, provenance

    def _fetch(self, grid: GridSpec | None = None) -> RasterData:
        if grid is None:
            return self._probe()
        return self._combine(grid)[0]

    # ------------------------------------------------------------------
    # Combining
    # ------------------------------------------------------------------

    def _combine(self, grid: GridSpec) -> tuple[RasterData, np.ndarray]:
        """Merge the sources onto *grid*, best source first."""
        array = np.full(grid.shape, np.nan, dtype=np.float32)
        provenance = np.full(grid.shape, -1, dtype=np.int16)
        uncovered = np.ones(grid.shape, dtype=bool)

        for index, source in enumerate(self._sources):
            if not uncovered.any():
                _log.info(
                    "Mosaic fully covered; not fetching %s",
                    ", ".join(f"'{n}'" for n in self._names[index:]),
                )
                break
            layer = self._fetch_layer(index, source, grid)
            if layer is None:
                continue
            fill = uncovered & ~np.isnan(layer)
            if not fill.any():
                continue
            array[fill] = layer[fill]
            provenance[fill] = index
            uncovered &= ~fill
            _log.info(
                "Mosaic: '%s' supplied %.1f%% of the grid",
                self._names[index], 100.0 * fill.sum() / fill.size,
            )

        return RasterData(array=array, crs=grid.crs, transform=grid.transform), provenance

    def _fetch_layer(
        self, index: int, source: DataSource, grid: GridSpec
    ) -> np.ndarray | None:
        """Fetch one source and align it to *grid*, or return None when it is skipped."""
        try:
            return align_to_grid(source.fetch(grid=grid), grid)
        except Exception as exc:
            if self._on_error == "raise":
                raise
            self._warn_skipped(index, exc)
            return None

    def _probe(self) -> RasterData:
        """Return the best source that can be fetched without a grid.

        This seeds the automatic grid, which needs a CRS and a pixel size
        before the sources can be aligned and merged.
        """
        if self.requires_grid:
            raise ValueError(_NEEDS_GRID)
        for index, source in enumerate(self._sources):
            if source.requires_grid:
                continue
            try:
                return source.fetch(grid=None)
            except Exception as exc:
                if self._on_error == "raise":
                    raise
                self._warn_skipped(index, exc)
        # Every candidate was skipped: report no spatial information, the way a
        # source without a CRS does, and leave the grid to another input.
        return RasterData(
            array=np.full((1, 1), np.nan, dtype=np.float32), crs=None, transform=None
        )

    def _warn_skipped(self, index: int, exc: Exception) -> None:
        warnings.warn(
            f"Skipping '{self._names[index]}' in the mosaic: "
            f"{type(exc).__name__}: {exc}",
            UserWarning,
            # The mosaic is reached from several call depths, so point at the
            # mosaic itself rather than at a frame that moves.
            stacklevel=2,
        )


_NEEDS_GRID = (
    "MosaicSource requires a grid context to combine its sources, which have "
    "to be aligned before they can be compared pixel by pixel.  This is "
    "provided automatically by GeoBayesianNetwork.infer() and by fetch_raw()."
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_sources(sources: Sequence[DataSource]) -> list[DataSource]:
    """Validate the source list."""
    if isinstance(sources, DataSource):
        raise ValueError(
            f"sources must be a list of data sources, not a single one; got {sources!r}"
        )
    if isinstance(sources, (str, bytes)) or not isinstance(sources, (list, tuple)):
        raise ValueError(f"sources must be a list of data sources; got {sources!r}")
    if not sources:
        raise ValueError("sources must hold at least one data source; got an empty list")
    for source in sources:
        if not isinstance(source, DataSource):
            raise ValueError(
                f"every entry in sources must be a data source; got {source!r}"
            )
    return list(sources)


def _check_names(names: Sequence[str] | None, sources: list[DataSource]) -> list[str]:
    """Validate ``names`` against the sources, or label them by class and position."""
    if names is None:
        return [f"{type(source).__name__}[{i}]" for i, source in enumerate(sources)]
    if isinstance(names, (str, bytes)) or not isinstance(names, (list, tuple)):
        raise ValueError(
            f"names must be a list of strings, one per source; got {names!r}"
        )
    if len(names) != len(sources):
        raise ValueError(
            f"names must hold one name per source; got {len(names)} name(s) for "
            f"{len(sources)} source(s)"
        )
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"names must be non-empty strings; got {name!r}")
    if len(set(names)) != len(names):
        raise ValueError(
            "names must be unique, so the provenance layer can be read back; "
            f"got {list(names)!r}"
        )
    return list(names)


def _check_on_error(value: str) -> str:
    """Validate the ``on_error`` policy."""
    if value not in _ON_ERROR:
        raise ValueError(
            f"on_error must be one of {', '.join(repr(v) for v in _ON_ERROR)}; "
            f"got {value!r}"
        )
    return value
