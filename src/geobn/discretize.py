"""Discretize continuous raster values into Bayesian network state indices."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiscretizationSpec:
    """Maps continuous values to BN state labels via breakpoints.

    Example
    -------
    ``DiscretizationSpec([0, 10, 30, 90], ["flat", "moderate", "steep"])``
    produces:
      - value < 10          → "flat"      (index 0)
      - 10 ≤ value < 30     → "moderate"  (index 1)
      - value ≥ 30          → "steep"     (index 2)

    The first and last breakpoints define the documented valid range but do
    not affect the bin boundaries used for digitization.
    """

    breakpoints: list[float]
    labels: list[str]

    def __post_init__(self) -> None:
        expected = len(self.breakpoints) - 1
        if len(self.labels) != expected:
            raise ValueError(
                f"Expected {expected} labels for {len(self.breakpoints)} breakpoints, "
                f"got {len(self.labels)}."
            )
        if len(self.breakpoints) < 2:
            raise ValueError("At least 2 breakpoints are required.")
        if any(b >= c for b, c in zip(self.breakpoints, self.breakpoints[1:])):
            raise ValueError("Breakpoints must be strictly increasing.")


def discretize_array(array: np.ndarray, spec: DiscretizationSpec) -> np.ndarray:
    """Return an integer index array (H, W) matching each pixel to a state.

    NaN pixels are mapped to -1 (sentinel for NoData).
    """
    # Interior bin edges (everything between first and last breakpoint)
    bins = spec.breakpoints[1:-1]
    indices = np.digitize(array, bins).astype(np.int16)

    # Mark NaN as -1
    nan_mask = np.isnan(array)
    indices[nan_mask] = -1

    return indices
