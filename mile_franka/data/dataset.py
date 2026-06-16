"""Builder for MILE intervention datasets (continuous / Box action spaces).

Accumulates per-timestep records and emits the exact dict schema that
mile/scripts/collect_synthetic_interventions.py produces and InterventionTrainer
consumes.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

BOX_KEYS = [
    "state",
    "rollout_action",
    "action",
    "intervention_prob",
    "intervention",
    "reward",
    "next_state",
    "done",
]


class InterventionDatasetBuilder:
    """Accumulate (s, a_r, a_h, nu, r, s', done) records into the MILE Box dict."""

    def __init__(self) -> None:
        self._d: Dict[str, List] = {k: [] for k in BOX_KEYS}

    def add(self, *, state, rollout_action, action, intervention,
            reward, next_state, done) -> None:
        nu = int(bool(intervention))
        self._d["state"].append(np.asarray(state, dtype=np.float32))
        self._d["rollout_action"].append(np.asarray(rollout_action, dtype=np.float32))
        self._d["action"].append(np.asarray(action, dtype=np.float32))
        self._d["intervention_prob"].append(
            np.array([1.0 - nu, float(nu)], dtype=np.float32)
        )
        self._d["intervention"].append(nu)
        self._d["reward"].append(float(reward))
        self._d["next_state"].append(np.asarray(next_state, dtype=np.float32))
        self._d["done"].append(bool(done))

    def to_dict(self) -> Dict[str, List]:
        """Return a shallow copy of the accumulated dict (lists per key)."""
        return {k: list(v) for k, v in self._d.items()}

    def __len__(self) -> int:
        return len(self._d["state"])
