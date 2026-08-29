from __future__ import annotations

from typing import Any

import numpy as np

from .data_contract import assert_graph_contract
from .schema import WTAInstance, WTASolution


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def instance_from_graph(graph: Any, dataset_version: str = "v5") -> WTAInstance:
    """Convert one current DeepWTA PyG graph into the solver-neutral input schema."""
    assert_graph_contract(graph)
    resource = getattr(graph, "scheme_resource", None)
    instance = WTAInstance(
        base_coords=_to_numpy(graph.base_coords).astype(np.float64, copy=True),
        target_coords=_to_numpy(graph.target_coords).astype(np.float64, copy=True),
        inventory=_to_numpy(graph.B).astype(np.float64, copy=True),
        scheme_demand=_to_numpy(graph.U).astype(np.float64, copy=True),
        damage_level=_to_numpy(graph.G).astype(np.float64, copy=True),
        damage_threshold=_to_numpy(graph.G_th).astype(np.float64, copy=True),
        intercept_prob=_to_numpy(graph.Q).astype(np.float64, copy=True),
        distance=_to_numpy(graph.D).astype(np.float64, copy=True),
        reachable=_to_numpy(graph.X).astype(np.float64, copy=True),
        instance_id=str(graph.scenario_id),
        seed=int(graph.scenario_seed),
        dataset_version=dataset_version,
        reference_scheme=_to_numpy(graph.reference_scheme).astype(np.int64, copy=True),
        reference_allocation=_to_numpy(graph.reference_allocation).astype(np.float64, copy=True),
        scheme_resource=(
            _to_numpy(resource).astype(np.float64, copy=True)
            if resource is not None
            else None
        ),
    )
    instance.check_shapes()
    return instance


def legacy_physical_kwargs(instance: WTAInstance) -> dict[str, np.ndarray]:
    """Names expected by the existing canonical validator/evaluator."""
    return {
        "B": instance.inventory,
        "U": instance.scheme_demand,
        "G": instance.damage_level,
        "G_th": instance.damage_threshold,
        "Q": instance.intercept_prob,
        "X": instance.reachable,
        "D": instance.distance,
        "base_coords": instance.base_coords,
        "target_coords": instance.target_coords,
    }


def solution_from_parts(
    selected_scheme: np.ndarray,
    allocation: np.ndarray,
    method: str = "",
    stage: str = "final",
) -> WTASolution:
    return WTASolution(
        selected_scheme=np.asarray(selected_scheme, dtype=np.int64).copy(),
        allocation=np.asarray(allocation).copy(),
        method=str(method),
        stage=str(stage),
    )
