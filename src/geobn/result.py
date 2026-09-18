"""InferenceResult — the object returned by GeoBayesianNetwork.infer()."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine

from ._io import write_geotiff
from .inference import shannon_entropy


@dataclass
class InferenceResult:
    """Holds per-pixel probability distributions for one or more query nodes.

    Attributes
    ----------
    probabilities:
        Mapping from query node name to a (H, W, n_states) float32 array.
        NaN where any input was NoData.
    state_names:
        Mapping from query node name to its ordered list of state labels.
    crs:
        CRS of the output grid as an EPSG string or WKT.
    transform:
        Affine pixel-to-world transform of the output grid.
    """

    probabilities: dict[str, np.ndarray]   # node → (H, W, n_states)
    state_names: dict[str, list[str]]
    crs: str
    transform: Affine

    # ------------------------------------------------------------------
    # Summary layers
    # ------------------------------------------------------------------
    #
    # Each returns a (H, W) float32 array, NaN where the probabilities are
    # NaN (NoData input).  State indices follow ``state_names[node]``.

    def entropy(self, node: str) -> np.ndarray:
        """Shannon entropy (bits) for *node*, shape (H, W)."""
        return shannon_entropy(self._probs(node))

    def expected_value(
        self, node: str, values: Sequence[float] | Mapping[str, float]
    ) -> np.ndarray:
        """Probability-weighted mean of *values*: sum of P(state) * value(state).

        Parameters
        ----------
        node:
            Query node name.
        values:
            A number per state, either as a sequence in state order
            (``[10, 50, 90]``) or as a mapping from state name to number
            (``{"low": 10, "medium": 50, "high": 90}``) covering every state.
            With consequence scores or costs this is the expected loss.
        """
        probs = self._probs(node)
        v = self._state_values(node, values)
        return (probs @ v).astype(np.float32)

    def std(
        self, node: str, values: Sequence[float] | Mapping[str, float]
    ) -> np.ndarray:
        """Standard deviation of *values* under the posterior.

        ``values`` is given as for :meth:`expected_value`.
        """
        probs = self._probs(node)
        v = self._state_values(node, values)
        mean = probs @ v
        var = (probs * (v - mean[..., np.newaxis]) ** 2).sum(axis=-1)
        return np.sqrt(np.maximum(var, 0.0)).astype(np.float32)

    def mode(self, node: str) -> np.ndarray:
        """Index of the most probable state (ties go to the lowest index)."""
        probs = self._probs(node)
        valid = self._valid_mask(probs)
        out = np.full(probs.shape[:2], np.nan, dtype=np.float32)
        out[valid] = np.argmax(probs[valid], axis=-1)
        return out

    def mode_probability(self, node: str) -> np.ndarray:
        """Probability of the most probable state."""
        return self._probs(node).max(axis=-1).astype(np.float32)

    def exceedance(self, node: str, state: str | int) -> np.ndarray:
        """P(node >= *state*): the probability of *state* plus every state
        after it in state order.

        *state* is a state name or index.  For a risk node ordered
        ``low, medium, high``, ``exceedance(node, "medium")`` is
        P(medium) + P(high).
        """
        probs = self._probs(node)
        k = self._state_index(node, state)
        return probs[..., k:].sum(axis=-1).astype(np.float32)

    def ignorance(self, node: str, threshold: float) -> np.ndarray:
        """Most probable state where its probability is at least *threshold*,
        NaN elsewhere (too uncertain to name a state).

        *threshold* must be in (0, 1].
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1]; got {threshold!r}")
        out = self.mode(node)
        out[~(self.mode_probability(node) >= threshold)] = np.nan
        return out

    def quantile(self, node: str, q: float) -> np.ndarray:
        """Index of the lowest state k with P(node <= k) >= *q*.

        With states ordered from least to most severe, ``quantile(node, 0.95)``
        is the worst state at 95 % confidence: the node is no worse than it
        with probability at least 0.95.  *q* must be in (0, 1].
        """
        if not 0.0 < q <= 1.0:
            raise ValueError(f"q must be in (0, 1]; got {q!r}")
        probs = self._probs(node)
        valid = self._valid_mask(probs)
        # Tolerance so float rounding (cumsum ending at 0.9999999) cannot
        # push q = 1 past the last state.
        reached = np.cumsum(probs[valid], axis=-1) >= q - 1e-6
        idx = np.where(reached.any(axis=-1), np.argmax(reached, axis=-1), probs.shape[-1] - 1)
        out = np.full(probs.shape[:2], np.nan, dtype=np.float32)
        out[valid] = idx
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probs(self, node: str) -> np.ndarray:
        if node not in self.probabilities:
            raise KeyError(
                f"{node!r} is not a query node of this result; "
                f"available: {list(self.probabilities)}"
            )
        return self.probabilities[node]

    @staticmethod
    def _valid_mask(probs: np.ndarray) -> np.ndarray:
        return np.all(np.isfinite(probs), axis=-1)

    def _state_values(
        self, node: str, values: Sequence[float] | Mapping[str, float]
    ) -> np.ndarray:
        states = self.state_names[node]
        if isinstance(values, Mapping):
            missing = [s for s in states if s not in values]
            unknown = [k for k in values if k not in states]
            if missing or unknown:
                raise ValueError(
                    f"values for {node!r} must have exactly the states {states}; "
                    f"missing {missing}, unknown {unknown}"
                )
            v = [values[s] for s in states]
        else:
            v = list(values)
            if len(v) != len(states):
                raise ValueError(
                    f"values for {node!r} must have {len(states)} entries "
                    f"(one per state {states}); got {len(v)}"
                )
        arr = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"values for {node!r} must be finite; got {v}")
        return arr

    def _state_index(self, node: str, state: str | int) -> int:
        states = self.state_names[node]
        if isinstance(state, str):
            if state not in states:
                raise ValueError(f"{state!r} is not a state of {node!r}; states: {states}")
            return states.index(state)
        if isinstance(state, (int, np.integer)) and not isinstance(state, bool):
            if not 0 <= state < len(states):
                raise ValueError(
                    f"state index {state} out of range for {node!r} "
                    f"({len(states)} states)"
                )
            return int(state)
        raise TypeError(f"state must be a state name or index; got {state!r}")

    def _grid_shape(self) -> tuple[int, int]:
        if not self.probabilities:
            raise ValueError("InferenceResult has no probability data")
        return next(iter(self.probabilities.values())).shape[:2]

    def _check_layers(
        self, layers: Mapping[str, np.ndarray], reserved: set[str]
    ) -> dict[str, np.ndarray]:
        shape = self._grid_shape()
        checked: dict[str, np.ndarray] = {}
        for name, arr in layers.items():
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise ValueError(
                    f"layer names must be non-empty strings without path "
                    f"separators; got {name!r}"
                )
            if name in reserved:
                raise ValueError(f"layer name {name!r} clashes with a query-node output")
            arr = np.asarray(arr)
            if arr.shape != shape:
                raise ValueError(
                    f"layer {name!r} has shape {arr.shape}; expected the grid "
                    f"shape {shape}"
                )
            checked[name] = arr.astype(np.float32)
        return checked

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_geotiff(
        self,
        output_dir: str | Path,
        layers: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        """Write one multi-band GeoTIFF per query node.

        Band layout (1-indexed):
          Bands 1…N  — P(state_i | evidence) for each state i
          Band N+1   — Shannon entropy

        Band descriptions contain the state label or "entropy".

        Parameters
        ----------
        output_dir:
            Directory to write into (created if needed).
        layers:
            Optional extra (H, W) layers, e.g. from :meth:`expected_value`.
            Each is written as a single-band ``{name}.tif`` on the same grid.
        """
        extra = self._check_layers(layers or {}, reserved=set(self.probabilities))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for node, probs in self.probabilities.items():
            H, W, n_states = probs.shape
            ent = self.entropy(node)[..., np.newaxis]  # (H, W, 1)
            cube = np.concatenate([probs, ent], axis=-1)  # (H, W, n_states+1)
            bands = cube.transpose(2, 0, 1)               # (bands, H, W)

            out_path = output_dir / f"{node}.tif"
            descriptions = [*self.state_names[node], "entropy"]
            write_geotiff(bands, self.crs, self.transform, out_path, descriptions=descriptions)

        for name, arr in extra.items():
            write_geotiff(
                arr[np.newaxis], self.crs, self.transform,
                output_dir / f"{name}.tif", descriptions=[name],
            )

    def show_map(
        self,
        output_dir: str | Path = ".",
        filename: str = "map.html",
        overlay_opacity: float = 0.65,
        open_browser: bool = True,
        extra_layers: dict[str, np.ndarray] | None = None,
        show_probability_bands: bool = True,
        show_category: bool = True,
        show_entropy: bool = True,
    ) -> Path:
        """Generate and optionally open an interactive Leaflet map.

        Parameters
        ----------
        output_dir:
            Directory to write the HTML file into.
        filename:
            Output filename (default ``map.html``).
        overlay_opacity:
            Opacity of probability overlays (0–1).
        open_browser:
            If True (default), open the map in the default browser.
        extra_layers:
            Additional named (H, W) arrays to include as overlays
            (e.g. ``{"Slope angle (°)": slope_deg}``).
        show_probability_bands:
            If False, omit the individual P(state) layers (default True).
        show_category:
            If False, omit the argmax category layer (default True).

        Returns
        -------
        Path
            Path to the written HTML file.
        """
        from ._viz import show_map as _show_map  # noqa: PLC0415

        return _show_map(
            result=self,
            output_dir=output_dir,
            filename=filename,
            overlay_opacity=overlay_opacity,
            open_browser=open_browser,
            extra_layers=extra_layers,
            show_probability_bands=show_probability_bands,
            show_category=show_category,
            show_entropy=show_entropy,
        )

    def to_xarray(self, layers: Mapping[str, np.ndarray] | None = None) -> xr.Dataset:
        """Return an xarray Dataset with spatial coordinates.

        Each query node becomes a DataArray with dimensions
        (state, y, x).  Entropy is added as a separate variable
        ``{node}_entropy`` with dimensions (y, x).

        Parameters
        ----------
        layers:
            Optional extra (H, W) layers, e.g. from :meth:`expected_value`,
            added as (y, x) variables under their names.
        """
        H, W = self._grid_shape()
        reserved = set(self.probabilities) | {f"{n}_entropy" for n in self.probabilities}
        extra = self._check_layers(layers or {}, reserved=reserved)
        transform = self.transform

        # Pixel-centre coordinates
        xs = transform.c + (np.arange(W) + 0.5) * transform.a
        ys = transform.f + (np.arange(H) + 0.5) * transform.e

        data_vars: dict[str, xr.DataArray] = {}

        for node, probs in self.probabilities.items():
            states = self.state_names[node]
            da = xr.DataArray(
                probs.transpose(2, 0, 1),  # (state, y, x)
                dims=["state", "y", "x"],
                coords={
                    "state": states,
                    "y": ys,
                    "x": xs,
                },
                name=node,
                attrs={"crs": self.crs},
            )
            data_vars[node] = da

            ent_da = xr.DataArray(
                self.entropy(node),
                dims=["y", "x"],
                coords={"y": ys, "x": xs},
                name=f"{node}_entropy",
                attrs={"crs": self.crs, "units": "bits"},
            )
            data_vars[f"{node}_entropy"] = ent_da

        for name, arr in extra.items():
            data_vars[name] = xr.DataArray(
                arr,
                dims=["y", "x"],
                coords={"y": ys, "x": xs},
                name=name,
                attrs={"crs": self.crs},
            )

        return xr.Dataset(data_vars)
