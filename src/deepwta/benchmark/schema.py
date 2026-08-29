from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WTAInstance:
    """Canonical physical WTA input shared by all solvers.

    `distance` is the actual D matrix stored by the current v5 dataset.  It is not
    renamed to flight time because the active data generator stores Euclidean route
    distance, and every baseline must receive the same information.
    """

    base_coords: np.ndarray                 # [N,2]
    target_coords: np.ndarray               # [M,2]
    inventory: np.ndarray                   # [K,N]
    scheme_demand: np.ndarray               # [L,K,M]
    damage_level: np.ndarray                # [L,M]
    damage_threshold: np.ndarray            # [M]
    intercept_prob: np.ndarray              # [K,N,M]
    distance: np.ndarray                    # [N,M]
    reachable: np.ndarray                   # [K,N,M]
    instance_id: str
    seed: int
    dataset_version: str = "v5"
    reference_scheme: np.ndarray | None = None       # [M], audit/benchmark only
    reference_allocation: np.ndarray | None = None   # [K,N,M], audit only
    scheme_resource: np.ndarray | None = None        # [L,M], audit only

    @property
    def num_weapons(self) -> int:
        return int(self.intercept_prob.shape[0])

    @property
    def num_bases(self) -> int:
        return int(self.intercept_prob.shape[1])

    @property
    def num_targets(self) -> int:
        return int(self.intercept_prob.shape[2])

    @property
    def num_schemes(self) -> int:
        return int(self.scheme_demand.shape[0])

    def check_shapes(self) -> None:
        k, n, m = self.intercept_prob.shape
        l = self.scheme_demand.shape[0]
        expected = {
            "base_coords": (n, 2),
            "target_coords": (m, 2),
            "inventory": (k, n),
            "scheme_demand": (l, k, m),
            "damage_level": (l, m),
            "damage_threshold": (m,),
            "distance": (n, m),
            "reachable": (k, n, m),
        }
        actual = {name: tuple(np.asarray(getattr(self, name)).shape) for name in expected}
        bad = {name: (actual[name], shape) for name, shape in expected.items() if actual[name] != shape}
        if bad:
            raise ValueError(f"WTAInstance shape mismatch: {bad}")
        if self.reference_scheme is not None and np.asarray(self.reference_scheme).shape != (m,):
            raise ValueError("reference_scheme must have shape [M]")
        if self.reference_allocation is not None and np.asarray(self.reference_allocation).shape != (k, n, m):
            raise ValueError("reference_allocation must have shape [K,N,M]")


@dataclass(frozen=True)
class WTASolution:
    selected_scheme: np.ndarray              # [M]
    allocation: np.ndarray                   # [K,N,M], integer for final evaluation
    method: str = ""
    stage: str = "final"


@dataclass(frozen=True)
class SolverStats:
    inference_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    total_time_ms: float = 0.0
    num_decisions: int = 0
    num_evaluations: int = 0


@dataclass(frozen=True)
class SolverResult:
    solution: WTASolution
    stats: SolverStats = field(default_factory=SolverStats)
    metadata: dict[str, Any] = field(default_factory=dict)
