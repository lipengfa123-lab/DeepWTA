from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class FrozenDataContract:
    """Read-only contract for the data already used by the current DeepWTA runs.

    This module intentionally does not regenerate or mutate the dataset.  Its role is
    to prevent future solvers from silently changing the problem.
    """

    experiment_revision: str = "v5"
    config_schema_revision: str = "v4"
    num_weapons: int = 10
    num_schemes: int = 10
    demand_max: int = 8
    inventory_max: int = 80
    reachability_rate: float = 0.97
    min_reachable_bases: int = 3
    train_size: int = 6000
    val_size: int = 800
    test_size: int = 800
    base_seed: int = 2026
    crossing_policy: str = "soft_tactical_objective"


FROZEN_DATA_CONTRACT = FrozenDataContract()

REQUIRED_GRAPH_FIELDS = (
    "B",
    "U",
    "G",
    "G_th",
    "Q",
    "X",
    "D",
    "base_coords",
    "target_coords",
    "reference_scheme",
    "reference_allocation",
    "scenario_id",
    "scenario_seed",
)


def _shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        return tuple(int(x) for x in value.shape)
    return tuple(np.asarray(value).shape)


def assert_graph_contract(graph: Any, contract: FrozenDataContract = FROZEN_DATA_CONTRACT) -> None:
    missing = [name for name in REQUIRED_GRAPH_FIELDS if not hasattr(graph, name)]
    if missing:
        raise ValueError(f"graph is missing frozen v5 fields: {missing}")

    q_shape = _shape(graph.Q)
    if len(q_shape) != 3:
        raise ValueError(f"Q must have shape [K,N,M], got {q_shape}")
    k, n, m = q_shape
    u_shape = _shape(graph.U)
    if k != contract.num_weapons:
        raise ValueError(f"expected K={contract.num_weapons}, got K={k}")
    if u_shape != (contract.num_schemes, k, m):
        raise ValueError(
            f"expected U shape [{contract.num_schemes},{k},{m}], got {u_shape}"
        )

    expected = {
        "B": (k, n),
        "G": (contract.num_schemes, m),
        "G_th": (m,),
        "X": (k, n, m),
        "D": (n, m),
        "base_coords": (n, 2),
        "target_coords": (m, 2),
        "reference_scheme": (m,),
        "reference_allocation": (k, n, m),
    }
    for name, shape in expected.items():
        actual = _shape(getattr(graph, name))
        if actual != shape:
            raise ValueError(f"{name} shape {actual} != {shape}")


def assert_manifest_contract(
    manifest: Mapping[str, Any],
    contract: FrozenDataContract = FROZEN_DATA_CONTRACT,
) -> None:
    """Best-effort audit for old/new manifest layouts without enforcing a version string.

    The historical package used different labels for the config schema and the
    manuscript experiment revision.  We therefore validate semantics, not a brittle
    literal version field.
    """

    cfg = manifest.get("config", manifest)
    wta = cfg.get("wta", cfg) if isinstance(cfg, Mapping) else {}
    split = cfg.get("split", {}) if isinstance(cfg, Mapping) else {}

    checks = {
        "num_weapons": contract.num_weapons,
        "num_schemes": contract.num_schemes,
        "demand_max": contract.demand_max,
        "inventory_max": contract.inventory_max,
    }
    for key, expected in checks.items():
        if key in wta and int(wta[key]) != expected:
            raise ValueError(f"manifest {key}={wta[key]} != frozen value {expected}")

    split_expected = {
        "train": contract.train_size,
        "val": contract.val_size,
        "test": contract.test_size,
    }
    split_counts = manifest.get("split_counts", {})
    for key, expected in split_expected.items():
        observed = split_counts.get(key, split.get(key))
        if observed is not None and int(observed) != expected:
            raise ValueError(f"manifest {key}={observed} != frozen value {expected}")
