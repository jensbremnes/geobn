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

    The first and last breakpoints define the valid range ``[first, last]``
    (inclusive).  They do not affect the bin boundaries; *out_of_range*
    controls what happens to values outside them:

      - ``"clip"`` (default): values below the range go to the first state,
        values above it go to the last state.
      - ``"nan"``: values outside the range are treated as NoData (index -1).
    """

    breakpoints: list[float]
    labels: list[str]
    out_of_range: str = "clip"

    def __post_init__(self) -> None:
        if self.out_of_range not in ("clip", "nan"):
            raise ValueError(
                f"out_of_range must be 'clip' or 'nan', got {self.out_of_range!r}."
            )
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

    NaN pixels are mapped to -1 (sentinel for NoData).  With
    ``spec.out_of_range == "nan"``, values outside
    ``[breakpoints[0], breakpoints[-1]]`` are mapped to -1 as well.
    """
    # Interior bin edges (everything between first and last breakpoint)
    bins = spec.breakpoints[1:-1]
    indices = np.digitize(array, bins).astype(np.int16)

    # Mark NaN as -1
    nan_mask = np.isnan(array)
    indices[nan_mask] = -1

    if spec.out_of_range == "nan":
        with np.errstate(invalid="ignore"):
            outside = (array < spec.breakpoints[0]) | (array > spec.breakpoints[-1])
        indices[outside] = -1

    return indices
