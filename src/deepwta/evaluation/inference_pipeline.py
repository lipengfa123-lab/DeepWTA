from __future__ import annotations

"""Small inference helpers shared by evaluation-only experiment programs."""

import numpy as np
import torch


def reconstruct_dense_allocation(graph, edge_values: torch.Tensor) -> np.ndarray:
    """Convert edge-wise model output [E,K] to the physical [K,N,M] tensor."""
    x = graph.x.detach().cpu()
    edge_index = graph.edge_index.detach().cpu()
    base_nodes = torch.nonzero(x[:, 2] < 0.5, as_tuple=False).reshape(-1)
    target_nodes = torch.nonzero(x[:, 2] >= 0.5, as_tuple=False).reshape(-1)
    base_lookup = torch.full((x.size(0),), -1, dtype=torch.long)
    target_lookup = torch.full((x.size(0),), -1, dtype=torch.long)
    base_lookup[base_nodes] = torch.arange(base_nodes.numel())
    target_lookup[target_nodes] = torch.arange(target_nodes.numel())
    base_id = base_lookup[edge_index[0]]
    target_id = target_lookup[edge_index[1]]
    if bool(torch.any(base_id < 0)) or bool(torch.any(target_id < 0)):
        raise ValueError("WTA edge_index must contain base-to-target edges")

    values = edge_values.detach().float().cpu()
    allocation = torch.zeros(
        (int(values.size(1)), int(base_nodes.numel()), int(target_nodes.numel())),
        dtype=values.dtype,
    )
    allocation[:, base_id, target_id] = values.T
    return allocation.numpy()


def select_level_feasible_schemes(
    scheme_probs: np.ndarray,
    damage_level: np.ndarray,
    damage_threshold: np.ndarray,
    tol: float = 1e-7,
) -> np.ndarray:
    """Select the highest-probability damage-level-valid scheme per target."""
    probs = np.asarray(scheme_probs, dtype=float)
    level = np.asarray(damage_level, dtype=float)
    threshold = np.asarray(damage_threshold, dtype=float)
    if probs.ndim != 2:
        raise ValueError("scheme_probs must have shape [M,L]")
    m_count, l_count = probs.shape
    if level.shape != (l_count, m_count) or threshold.shape != (m_count,):
        raise ValueError("scheme probability and damage tensor shape mismatch")

    selected = np.empty(m_count, dtype=np.int64)
    for m in range(m_count):
        valid = np.flatnonzero(level[:, m] >= threshold[m] - tol)
        if valid.size == 0:
            raise ValueError(f"target {m} has no damage-level-feasible scheme")
        selected[m] = int(valid[np.argmax(probs[m, valid])])
    return selected

