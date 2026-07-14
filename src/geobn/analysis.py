"""Sensitivity analysis and value of information for decision support.

Given a *baseline* evidence profile (the conditions you expect), rank the
evidence nodes by how much they matter for a target posterior:

- **Tornado swing** (one-way sensitivity): vary one evidence node over its
  states with the rest of the baseline fixed; the swing is the max − min
  of ``P(target = target_state)``. The classic tornado-diagram input.
- **Value of information**: the expected reduction in Shannon entropy of
  the target from observing the node, weighted by the node's predictive
  distribution given the *rest* of the baseline. Pre-decision, the
  "baseline observation" of a node is typically a forecast — VOI ranks
  which forecast is most worth verifying with a real measurement.

Example
-------
>>> import geobn
>>> from geobn.analysis import tornado
>>> bn = geobn.load("fire_risk.bif")
>>> baseline = {"slope": "steep", "rainfall": "low"}
>>> for entry in tornado(bn, "fire_risk", "high", baseline):
...     print(entry.node, entry.swing, entry.voi_bits)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

_log = logging.getLogger(__name__)

__all__ = ["TornadoEntry", "tornado"]


@dataclass(frozen=True)
class TornadoEntry:
    """One-way sensitivity result for a single evidence node."""

    node: str
    #: max - min of P(target = target_state) across this node's states
    swing: float
    #: P(target = target_state) for each state of this node
    p_by_state: dict[str, float]
    #: the node's state in the supplied baseline
    baseline_state: str
    #: expected Shannon-entropy reduction of the target (bits) from
    #: observing this node, given the rest of the baseline
    voi_bits: float


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _resolve_model(bn) -> DiscreteBayesianNetwork:
    """Accept either a GeoBayesianNetwork or a raw pgmpy network."""
    if isinstance(bn, DiscreteBayesianNetwork):
        return bn
    model = getattr(bn, "_model", None)
    if isinstance(model, DiscreteBayesianNetwork):
        return model
    raise TypeError(
        f"Expected GeoBayesianNetwork or DiscreteBayesianNetwork, got {type(bn).__name__}"
    )


def tornado(
    bn,
    target: str,
    target_state: str,
    baseline: dict[str, str],
) -> list[TornadoEntry]:
    """One-way sensitivity + value of information for each baseline node.

    Parameters
    ----------
    bn:
        A :class:`~geobn.GeoBayesianNetwork` or a pgmpy
        ``DiscreteBayesianNetwork``.
    target:
        The query node whose posterior is analysed.
    target_state:
        The state of *target* whose probability defines the swing.
    baseline:
        ``{evidence_node: state_name}`` — the reference conditions. Each
        node in the baseline is varied in turn while the others stay fixed.

    Returns
    -------
    list[TornadoEntry]
        Sorted by decreasing swing.
    """
    model = _resolve_model(bn)
    state_names = {
        cpd.variable: list(cpd.state_names[cpd.variable]) for cpd in model.get_cpds()
    }
    if target not in state_names:
        raise ValueError(f"Unknown target node '{target}'")
    if target_state not in state_names[target]:
        raise ValueError(
            f"Unknown state '{target_state}' for '{target}'; "
            f"expected one of {state_names[target]}"
        )
    for node, state in baseline.items():
        if node not in state_names:
            raise ValueError(f"Unknown baseline node '{node}'")
        if state not in state_names[node]:
            raise ValueError(
                f"Unknown state '{state}' for '{node}'; expected one of {state_names[node]}"
            )
        if node == target:
            raise ValueError(f"Target '{target}' cannot be part of the baseline evidence")

    ve = VariableElimination(model)
    t_idx = state_names[target].index(target_state)

    entries: list[TornadoEntry] = []
    for node, base_state in baseline.items():
        rest = {k: v for k, v in baseline.items() if k != node}

        p_by_state: dict[str, float] = {}
        posteriors: list[np.ndarray] = []
        for state in state_names[node]:
            post = ve.query(
                [target], evidence={**rest, node: state}, show_progress=False
            ).values
            p_by_state[state] = float(post[t_idx])
            posteriors.append(post)

        # VOI: predictive distribution of `node` given the rest of the baseline
        predictive = ve.query([node], evidence=rest, show_progress=False).values
        h_prior = _entropy_bits(ve.query([target], evidence=rest, show_progress=False).values)
        h_expected = float(
            sum(w * _entropy_bits(post) for w, post in zip(predictive, posteriors))
        )

        entries.append(
            TornadoEntry(
                node=node,
                swing=max(p_by_state.values()) - min(p_by_state.values()),
                p_by_state=p_by_state,
                baseline_state=base_state,
                voi_bits=max(h_prior - h_expected, 0.0),
            )
        )

    entries.sort(key=lambda e: e.swing, reverse=True)
    _log.info(
        "Tornado on %s=%s over %d nodes; top driver: %s (swing %.3g)",
        target,
        target_state,
        len(entries),
        entries[0].node if entries else "n/a",
        entries[0].swing if entries else 0.0,
    )
    return entries
