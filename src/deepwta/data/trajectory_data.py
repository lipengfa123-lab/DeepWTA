from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from deepwta.data.map_data import MapContext, sample_coordinates


@dataclass(frozen=True)
class PruningStats:
    total_pairs: int
    shared_endpoint_pruned: int
    x_axis_pruned: int
    y_axis_pruned: int
    candidate_pairs: int
    kept_pairs: int


def _as_xy(coords_rc: np.ndarray) -> np.ndarray:
    return np.stack([coords_rc[:, 1], coords_rc[:, 0]], axis=1).astype(np.float64)


def build_trajectory_features(
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    routes: np.ndarray,
    map_height: int,
    map_width: int,
) -> np.ndarray:
    """Build 7-D node features for directed 2D base->target routes."""
    base_xy = _as_xy(base_coords)
    target_xy = _as_xy(target_coords)
    start = base_xy[routes[:, 0]]
    end = target_xy[routes[:, 1]]

    vector = end - start
    length = np.linalg.norm(vector, axis=1)
    safe_length = np.maximum(length, 1e-9)
    direction = vector / safe_length[:, None]
    diagonal = max(float(np.hypot(map_height, map_width)), 1.0)

    return np.column_stack([
        length / diagonal,
        direction[:, 0],
        direction[:, 1],
        start[:, 0] / map_width,
        start[:, 1] / map_height,
        end[:, 0] / map_width,
        end[:, 1] / map_height,
    ]).astype(np.float32)


def _route_geometry(
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    routes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_xy = _as_xy(base_coords)
    target_xy = _as_xy(target_coords)
    start = base_xy[routes[:, 0]]
    end = target_xy[routes[:, 1]]
    min_xy = np.minimum(start, end)
    max_xy = np.maximum(start, end)
    return start, end, min_xy, max_xy


def broad_phase_filter(
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    routes: np.ndarray,
) -> tuple[np.ndarray, PruningStats]:
    """
    Sweep-and-Prune + AABB pruning.

    This is deterministic candidate pruning, not neural-network weight pruning.
    It removes impossible route pairs before the Conflict GNN is called.
    """
    route_count = len(routes)
    if route_count < 2:
        stats = PruningStats(0, 0, 0, 0, 0, 0)
        return np.empty((0, 2), dtype=np.int64), stats

    _, _, min_xy, max_xy = _route_geometry(base_coords, target_coords, routes)
    total_pairs = route_count * (route_count - 1) // 2

    order = np.argsort(min_xy[:, 0], kind="stable")
    candidates: list[tuple[int, int]] = []
    shared_pruned = 0
    x_pruned = 0
    y_pruned = 0

    for pos, i in enumerate(order):
        max_x_i = max_xy[i, 0]
        for next_pos in range(pos + 1, len(order)):
            j = order[next_pos]
            if min_xy[j, 0] > max_x_i:
                # Because min_x is sorted, all following pairs with i are impossible.
                x_pruned += len(order) - next_pos
                break

            if routes[i, 0] == routes[j, 0] or routes[i, 1] == routes[j, 1]:
                shared_pruned += 1
                continue

            if max_xy[i, 1] < min_xy[j, 1] or max_xy[j, 1] < min_xy[i, 1]:
                y_pruned += 1
                continue
            candidates.append((int(i), int(j)))

    candidate_array = np.asarray(candidates, dtype=np.int64).reshape(-1, 2)
    stats = PruningStats(
        total_pairs=total_pairs,
        shared_endpoint_pruned=shared_pruned,
        x_axis_pruned=x_pruned,
        y_axis_pruned=y_pruned,
        candidate_pairs=len(candidate_array),
        kept_pairs=len(candidate_array),
    )
    return candidate_array, stats


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _sign(value: float, eps: float = 1e-10) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _positive_collinear_overlap(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float) -> bool:
    if abs(b[0] - a[0]) >= abs(b[1] - a[1]):
        left = max(min(a[0], b[0]), min(c[0], d[0]))
        right = min(max(a[0], b[0]), max(c[0], d[0]))
    else:
        left = max(min(a[1], b[1]), min(c[1], d[1]))
        right = min(max(a[1], b[1]), max(c[1], d[1]))
    return (right - left) > eps


def compute_exact_cross_label(
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    routes: np.ndarray,
    pairs: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """1 = interior 2D crossing or positive collinear overlap; endpoint touch is 0."""
    start, end, _, _ = _route_geometry(base_coords, target_coords, routes)
    labels = np.zeros(len(pairs), dtype=np.int64)

    for idx, (i, j) in enumerate(pairs):
        # Shared start/end is excluded by definition even if geometry touches.
        if routes[i, 0] == routes[j, 0] or routes[i, 1] == routes[j, 1]:
            continue

        a, b, c, d = start[i], end[i], start[j], end[j]
        s1 = _sign(_orientation(a, b, c), eps)
        s2 = _sign(_orientation(a, b, d), eps)
        s3 = _sign(_orientation(c, d, a), eps)
        s4 = _sign(_orientation(c, d, b), eps)

        if s1 * s2 < 0 and s3 * s4 < 0:
            labels[idx] = 1
        elif s1 == s2 == s3 == s4 == 0 and _positive_collinear_overlap(a, b, c, d, eps):
            labels[idx] = 1
    return labels


def build_dual_edge_features(
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    routes: np.ndarray,
    pairs: np.ndarray,
    map_height: int,
    map_width: int,
) -> np.ndarray:
    start, end, _, _ = _route_geometry(base_coords, target_coords, routes)
    vector = end - start
    length = np.linalg.norm(vector, axis=1)
    safe_length = np.maximum(length, 1e-9)
    direction = vector / safe_length[:, None]
    midpoint = (start + end) * 0.5
    diagonal = max(float(np.hypot(map_height, map_width)), 1.0)

    i = pairs[:, 0]
    j = pairs[:, 1]
    rel_mid = (midpoint[j] - midpoint[i]) / np.array([map_width, map_height], dtype=np.float64)
    mid_dist = np.linalg.norm(midpoint[j] - midpoint[i], axis=1) / diagonal

    dot = np.sum(direction[i] * direction[j], axis=1)
    cross = direction[i, 0] * direction[j, 1] - direction[i, 1] * direction[j, 0]
    ratio = np.minimum(length[i], length[j]) / np.maximum(np.maximum(length[i], length[j]), 1e-9)

    return np.column_stack([
        rel_mid[:, 0],
        rel_mid[:, 1],
        mid_dist,
        np.clip(dot, -1.0, 1.0),
        np.clip(cross, -1.0, 1.0),
        ratio,
    ]).astype(np.float32)


def _sample_routes(
    num_bases: int,
    num_targets: int,
    route_count: int,
    reachability_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    all_routes = np.stack(
        np.meshgrid(np.arange(num_bases), np.arange(num_targets), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    valid = rng.random(len(all_routes)) < reachability_rate
    usable = all_routes[valid]
    if len(usable) < route_count:
        usable = all_routes
    indices = rng.choice(len(usable), size=min(route_count, len(usable)), replace=False)
    return usable[indices].astype(np.int64)


def _balance_candidate_pairs(
    pairs: np.ndarray,
    labels: np.ndarray,
    negative_to_positive_ratio: float,
    max_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    if len(pos) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)

    max_neg = int(np.ceil(len(pos) * negative_to_positive_ratio))
    if len(neg) > max_neg:
        neg = rng.choice(neg, size=max_neg, replace=False)

    keep = np.concatenate([pos, neg])
    if len(keep) > max_pairs:
        # Preserve all positives when possible; cap negatives first.
        pos_keep = pos[: min(len(pos), max_pairs)]
        remaining = max_pairs - len(pos_keep)
        neg_keep = neg[: max(0, remaining)]
        keep = np.concatenate([pos_keep, neg_keep])

    rng.shuffle(keep)
    return pairs[keep], labels[keep]



def _sample_graph_size(cfg, context: MapContext, rng: np.random.Generator) -> tuple[int, int]:
    """Sample a surrogate scenario under min_ratio <= N/M <= max_ratio."""
    min_ratio = float(getattr(cfg, "min_size_ratio", 1.0 / 3.0))
    max_ratio = float(getattr(cfg, "max_size_ratio", 3.0))
    if not (0.0 < min_ratio <= max_ratio):
        raise ValueError(
            f"invalid N/M ratio range: min_size_ratio={min_ratio}, "
            f"max_size_ratio={max_ratio}"
        )

    max_n = min(int(cfg.max_bases), len(context.base_candidates))
    max_m = min(int(cfg.max_targets), len(context.target_candidates))
    min_n = min(int(cfg.min_bases), max_n)
    min_m_cfg = min(int(cfg.min_targets), max_m)

    for _ in range(64):
        n = int(rng.integers(min_n, max_n + 1))
        min_m = max(min_m_cfg, int(np.ceil(n / max_ratio)))
        max_m_for_n = min(max_m, int(np.floor(n / min_ratio)))
        if min_m <= max_m_for_n:
            m = int(rng.integers(min_m, max_m_for_n + 1))
            ratio = n / m
            if min_ratio - 1e-12 <= ratio <= max_ratio + 1e-12:
                return n, m

    raise RuntimeError(
        "cannot sample a valid surrogate N/M pair; "
        f"requested ratio=[{min_ratio:.4f}, {max_ratio:.4f}], "
        f"N=[{min_n}, {max_n}], M=[{min_m_cfg}, {max_m}]"
    )



def generate_conflict_graph(cfg, context: MapContext, seed: int, scenario_id: str) -> Data:
    rng = np.random.default_rng(seed)

    for _ in range(cfg.max_graph_attempts):
        num_bases, num_targets = _sample_graph_size(cfg, context, rng)
        base_coords = sample_coordinates(context.base_candidates, num_bases, rng)
        target_coords = sample_coordinates(context.target_candidates, num_targets, rng)

        route_count = int(rng.integers(cfg.min_trajectories, cfg.max_trajectories + 1))
        routes = _sample_routes(num_bases, num_targets, route_count, cfg.route_reachability_rate, rng)
        if len(routes) < 2:
            continue

        pairs, stats = broad_phase_filter(base_coords, target_coords, routes)
        if len(pairs) == 0:
            continue
        labels = compute_exact_cross_label(base_coords, target_coords, routes, pairs)
        pairs, labels = _balance_candidate_pairs(
            pairs, labels, cfg.negative_to_positive_ratio, cfg.max_candidate_pairs, rng
        )
        if len(pairs) == 0 or int(np.sum(labels)) == 0:
            continue

        node_x = build_trajectory_features(
            base_coords, target_coords, routes, context.height, context.width
        )
        edge_attr_undirected = build_dual_edge_features(
            base_coords, target_coords, routes, pairs, context.height, context.width
        )

        # GAT receives bidirectional dual edges; labels are duplicated consistently.
        reverse_pairs = pairs[:, ::-1]
        reverse_edge_attr = build_dual_edge_features(
            base_coords, target_coords, routes, reverse_pairs, context.height, context.width
        )
        edge_index = np.concatenate([pairs.T, reverse_pairs.T], axis=1)
        edge_attr = np.concatenate([edge_attr_undirected, reverse_edge_attr], axis=0)
        edge_y = np.concatenate([labels, labels], axis=0)

        data = Data(
            x=torch.from_numpy(node_x),
            edge_index=torch.from_numpy(edge_index.astype(np.int64)),
            edge_attr=torch.from_numpy(edge_attr),
            edge_y=torch.from_numpy(edge_y.astype(np.int64)),
            y=torch.tensor([int(np.sum(labels))], dtype=torch.float32),
        )
        data.routes = torch.from_numpy(routes.astype(np.int64))
        data.base_coords = torch.from_numpy(base_coords.astype(np.float32))
        data.target_coords = torch.from_numpy(target_coords.astype(np.float32))
        data.undirected_pairs = torch.from_numpy(pairs.astype(np.int64))
        data.undirected_edge_y = torch.from_numpy(labels.astype(np.int64))
        data.total_pairs_before_pruning = int(stats.total_pairs)
        data.candidate_pairs_after_aabb = int(stats.candidate_pairs)
        data.pairs_after_sampling = int(len(pairs))
        data.scenario_seed = int(seed)
        data.scenario_id = str(scenario_id)
        return data

    raise RuntimeError(f"failed to generate a conflict graph after {cfg.max_graph_attempts} attempts")
