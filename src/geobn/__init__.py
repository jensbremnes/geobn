"""geobn — Bayesian network inference over geographic space."""
import logging as _logging

from .network import GeoBayesianNetwork, load
from .result import InferenceResult


def set_verbose(enabled: bool = True) -> None:
    """Enable or disable INFO-level logging for all geobn modules.

    Call ``geobn.set_verbose()`` at the top of your script to see what the
    library is doing at runtime — cache hits/misses, HTTP requests,
    reprojection, inference batching, etc.

    Parameters
    ----------
    enabled:
        ``True`` (default) sets the geobn logger to INFO and attaches a
        stderr handler if none is present.  ``False`` silences it again.
    """
    logger = _logging.getLogger("geobn")
    if enabled:
        logger.setLevel(_logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = _logging.StreamHandler()
            handler.setFormatter(_logging.Formatter("[geobn] %(message)s"))
            logger.addHandler(handler)
    else:
        logger.setLevel(_logging.WARNING)
        logger.propagate = True


from .sources import (
    ArraySource,
    ConstantSource,
    PointGridSource,
    RasterSource,
    URLSource,
    WCSSource,
)

__all__ = [
    "load",
    "set_verbose",
    "GeoBayesianNetwork",
    "InferenceResult",
    "ArraySource",
    "ConstantSource",
    "PointGridSource",
    "RasterSource",
    "URLSource",
    "WCSSource",
]
