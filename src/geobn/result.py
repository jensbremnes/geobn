"""InferenceResult — the object returned by GeoBayesianNetwork.infer()."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    # Derived property
    # ------------------------------------------------------------------

    def entropy(self, node: str) -> np.ndarray:
        """Shannon entropy (bits) for *node*, shape (H, W)."""
        return shannon_entropy(self.probabilities[node])

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_geotiff(self, output_dir: str | Path) -> None:
        """Write one multi-band GeoTIFF per query node.

        Band layout (1-indexed):
          Bands 1…N  — P(state_i | evidence) for each state i
          Band N+1   — Shannon entropy

        Band descriptions contain the state label or "entropy".
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for node, probs in self.probabilities.items():
            H, W, n_states = probs.shape
            ent = self.entropy(node)[..., np.newaxis]  # (H, W, 1)
            cube = np.concatenate([probs, ent], axis=-1)  # (H, W, n_states+1)
            bands = cube.transpose(2, 0, 1)               # (bands, H, W)

            out_path = output_dir / f"{node}.tif"
            write_geotiff(bands, self.crs, self.transform, out_path)

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
        score_threshold: float = 0.0,
    ) -> Path:
        """Generate and optionally open an interactive Leaflet map.

        Requires ``folium`` (``pip install geobn[viz]``).

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

        return _show_map(self, output_dir, filename, overlay_opacity,
                         open_browser, extra_layers,
                         show_probability_bands, show_category,
                         show_entropy, score_threshold)

    def to_xarray(self) -> "xr.Dataset":
        """Return an xarray Dataset with spatial coordinates.

        Each query node becomes a DataArray with dimensions
        (state, y, x).  Entropy is added as a separate variable
        ``{node}_entropy`` with dimensions (y, x).

        Requires ``xarray`` (``pip install "geobn[full]"``).
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "xarray is required for to_xarray(). "
                'Install it with: pip install "geobn[full]"'
            ) from exc

        H, W = next(iter(self.probabilities.values())).shape[:2]
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

        return xr.Dataset(data_vars)
