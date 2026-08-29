from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from deepwta.utils.env import add_project_paths  # noqa: E402
PROJECT_ROOT = add_project_paths()

from configs.dataset_config import CONFIG
from deepwta.data.map_data import pairwise_distance
from deepwta.data.trajectory_data import compute_exact_cross_label


def iter_graphs(dataset_dir: Path, split: str, limit: int | None):
    count = 0
    for shard in sorted((dataset_dir / split).glob("part_*.pt")):
        graphs = torch.load(shard, map_location="cpu", weights_only=False)
        for graph in graphs:
            yield graph
            count += 1
            if limit is not None and count >= limit:
                return


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_wta_graph(graph) -> list[str]:
    errors: list[str] = []

    k = int(graph.num_weapons)
    n = int(graph.num_bases)
    m = int(graph.num_targets)
    l = int(graph.num_schemes)

    size_ratio = n / m
    if not (CONFIG.wta.min_size_ratio - 1e-12 <= size_ratio <= CONFIG.wta.max_size_ratio + 1e-12):
        errors.append(
            f"N/M ratio {size_ratio:.4f} outside "
            f"[{CONFIG.wta.min_size_ratio:.4f}, {CONFIG.wta.max_size_ratio:.4f}]"
        )

    b = _as_numpy(graph.B)
    u = _as_numpy(graph.U)
    g = _as_numpy(graph.G)
    g_th = _as_numpy(graph.G_th)
    q = _as_numpy(graph.Q)
    x = _as_numpy(graph.X)
    d = _as_numpy(graph.D)
    base_coords = _as_numpy(graph.base_coords)
    target_coords = _as_numpy(graph.target_coords)
    selected = _as_numpy(graph.reference_scheme).astype(np.int64)
    a_ref = _as_numpy(graph.reference_allocation)
    resource = _as_numpy(graph.scheme_resource)

    expected_shapes = {
        "B": (k, n),
        "U": (l, k, m),
        "G": (l, m),
        "G_th": (m,),
        "Q": (k, n, m),
        "X": (k, n, m),
        "D": (n, m),
        "base_coords": (n, 2),
        "target_coords": (m, 2),
        "reference_scheme": (m,),
        "reference_allocation": (k, n, m),
        "scheme_resource": (l, m),
    }
    values = {
        "B": b, "U": u, "G": g, "G_th": g_th, "Q": q, "X": x, "D": d,
        "base_coords": base_coords, "target_coords": target_coords,
        "reference_scheme": selected, "reference_allocation": a_ref,
        "scheme_resource": resource,
    }
    for name, shape in expected_shapes.items():
        if values[name].shape != shape:
            errors.append(f"{name}: shape {values[name].shape} != {shape}")

    node_dim = 4 + k + l * k + l
    edge_dim = 1 + 2 * k
    if tuple(graph.x.shape) != (n + m, node_dim):
        errors.append(f"x: shape {tuple(graph.x.shape)} != {(n + m, node_dim)}")
    if tuple(graph.edge_attr.shape) != (n * m, edge_dim):
        errors.append(f"edge_attr: shape {tuple(graph.edge_attr.shape)} != {(n * m, edge_dim)}")
    if tuple(graph.edge_index.shape) != (2, n * m):
        errors.append(f"edge_index: shape {tuple(graph.edge_index.shape)} != {(2, n * m)}")

    for name, value in values.items():
        if name == "reference_scheme":
            continue
        if not np.all(np.isfinite(value)):
            errors.append(f"{name}: contains non-finite values")

    if np.any((q < 0.0) | (q > 1.0)):
        errors.append("Q: outside [0,1]")
    if np.any((x != 0.0) & (x != 1.0)):
        errors.append("X: not binary")
    if np.any(b < 0) or np.any(b > CONFIG.wta.inventory_max):
        errors.append("B: outside configured inventory range")
    if np.any(u < 0) or np.any(u > CONFIG.wta.demand_max):
        errors.append("U: outside configured demand range")
    if np.any(~np.isin(g, [1, 2, 3])):
        errors.append("G: contains values outside {1,2,3}")
    if np.any(~np.isin(g_th, [1, 2, 3])):
        errors.append("G_th: contains values outside {1,2,3}")

    recomputed_d = pairwise_distance(base_coords, target_coords)
    if not np.allclose(d, recomputed_d, atol=1e-4):
        errors.append("D: inconsistent with coordinates")

    required_reachable = min(CONFIG.wta.min_reachable_bases, n)
    reachable_count = np.sum(x > 0.5, axis=1)
    if np.any(reachable_count < required_reachable):
        errors.append("X: some weapon-target pairs have too few reachable bases")

    for target in range(m):
        levels = set(int(v) for v in g[:, target])
        if not {1, 2, 3}.issubset(levels):
            errors.append(f"target {target}: missing Light/Medium/Heavy scheme")
            break

        means = [
            float(np.mean(resource[g[:, target] == level, target]))
            for level in (1, 2, 3)
        ]
        gap = CONFIG.wta.level_resource_gap
        if means[1] + 1e-6 < means[0] * (1.0 + gap):
            errors.append(f"target {target}: Medium resource hierarchy failed")
            break
        if means[2] + 1e-6 < means[1] * (1.0 + gap):
            errors.append(f"target {target}: Heavy resource hierarchy failed")
            break

    if np.any(selected < 0) or np.any(selected >= l):
        errors.append("reference_scheme: invalid index")
    else:
        if np.any(g[selected, np.arange(m)] < g_th):
            errors.append("reference_scheme: violates G_th")

        if np.any(a_ref < 0) or not np.allclose(a_ref, np.rint(a_ref), atol=1e-6):
            errors.append("reference_allocation: not non-negative integer")
        if np.any(np.sum(a_ref, axis=2) > b + 1e-6):
            errors.append("reference_allocation: exceeds B")
        if np.any((a_ref > 0) & (x < 0.5)):
            errors.append("reference_allocation: uses X=0 route")

        chosen_demand = u[selected, :, np.arange(m)].T
        effective_hits = np.sum(a_ref * (1.0 - q), axis=1)
        if np.any(effective_hits + 1e-6 < chosen_demand):
            errors.append("reference_allocation: post-interception demand not satisfied")

    return errors


def validate_conflict_graph(graph) -> list[str]:
    errors: list[str] = []

    x = _as_numpy(graph.x)
    edge_index = _as_numpy(graph.edge_index).astype(np.int64)
    edge_attr = _as_numpy(graph.edge_attr)
    edge_y = _as_numpy(graph.edge_y).astype(np.int64)
    routes = _as_numpy(graph.routes).astype(np.int64)
    base_coords = _as_numpy(graph.base_coords)
    target_coords = _as_numpy(graph.target_coords)
    size_ratio = len(base_coords) / len(target_coords)
    if not (
        CONFIG.conflict.min_size_ratio - 1e-12
        <= size_ratio
        <= CONFIG.conflict.max_size_ratio + 1e-12
    ):
        errors.append(
            f"N/M ratio {size_ratio:.4f} outside "
            f"[{CONFIG.conflict.min_size_ratio:.4f}, {CONFIG.conflict.max_size_ratio:.4f}]"
        )
    pairs = _as_numpy(graph.undirected_pairs).astype(np.int64)
    labels = _as_numpy(graph.undirected_edge_y).astype(np.int64)

    if x.ndim != 2 or x.shape[1] != CONFIG.conflict.node_dim:
        errors.append(f"x: invalid shape {x.shape}")
    if edge_attr.ndim != 2 or edge_attr.shape[1] != CONFIG.conflict.edge_dim:
        errors.append(f"edge_attr: invalid shape {edge_attr.shape}")
    if edge_index.shape != (2, 2 * len(pairs)):
        errors.append("edge_index: bidirectional pair count mismatch")
    if len(edge_y) != 2 * len(labels):
        errors.append("edge_y: bidirectional label count mismatch")
    if not np.all(np.isin(labels, [0, 1])):
        errors.append("undirected_edge_y: labels must be 0/1")
    if not np.all(np.isin(edge_y, [0, 1])):
        errors.append("edge_y: labels must be 0/1")

    if len(pairs) > 0:
        if np.any(pairs[:, 0] == pairs[:, 1]):
            errors.append("undirected_pairs: contains self-loop")
        if np.any(routes[pairs[:, 0], 0] == routes[pairs[:, 1], 0]):
            errors.append("undirected_pairs: contains shared-start pair")
        if np.any(routes[pairs[:, 0], 1] == routes[pairs[:, 1], 1]):
            errors.append("undirected_pairs: contains shared-end pair")

        recomputed = compute_exact_cross_label(base_coords, target_coords, routes, pairs)
        if not np.array_equal(recomputed.astype(np.int64), labels):
            errors.append("undirected_edge_y: inconsistent with exact 2D geometry")

        if not np.array_equal(edge_y[:len(labels)], labels):
            errors.append("edge_y: forward labels inconsistent")
        if not np.array_equal(edge_y[len(labels):], labels):
            errors.append("edge_y: reverse labels inconsistent")

    expected_cross = int(np.sum(labels))
    actual_cross = int(round(float(_as_numpy(graph.y).reshape(-1)[0])))
    if expected_cross != actual_cross:
        errors.append(f"y: {actual_cross} != exact crossing count {expected_cross}")

    before = int(graph.total_pairs_before_pruning)
    after = int(graph.candidate_pairs_after_aabb)
    sampled = int(graph.pairs_after_sampling)
    if before < after or after < sampled:
        errors.append("pair counts: pruning/sampling counters are inconsistent")

    return errors


def validate_split(kind: str, split: str, limit: int | None) -> tuple[set[str], set[int], int, int]:
    dataset_dir = CONFIG.processed_dir / f"{kind}_{CONFIG.version}"
    validator = validate_wta_graph if kind == "wta" else validate_conflict_graph

    ids: set[str] = set()
    seeds: set[int] = set()
    checked = 0
    failed = 0

    for graph in iter_graphs(dataset_dir, split, limit):
        checked += 1
        scenario_id = str(graph.scenario_id)
        seed = int(graph.scenario_seed)

        if scenario_id in ids:
            print(f"[FAIL] {split}: duplicate scenario_id={scenario_id}")
            failed += 1
        if seed in seeds:
            print(f"[FAIL] {split}: duplicate seed={seed}")
            failed += 1

        ids.add(scenario_id)
        seeds.add(seed)

        errors = validator(graph)
        if errors:
            failed += 1
            print(f"[FAIL] {split}/{scenario_id}: " + "; ".join(errors[:6]))

    print(f"[{split}] checked={checked}, failed={failed}")
    return ids, seeds, checked, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict post-generation dataset audit")
    parser.add_argument("kind", choices=["wta", "conflict"])
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="0 = validate every graph")
    args = parser.parse_args()

    limit = None if args.limit <= 0 else args.limit
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    split_ids: dict[str, set[str]] = {}
    split_seeds: dict[str, set[int]] = {}
    total_failed = 0

    for split in splits:
        ids, seeds, _, failed = validate_split(args.kind, split, limit)
        split_ids[split] = ids
        split_seeds[split] = seeds
        total_failed += failed

    if len(splits) == 3 and limit is None:
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            id_overlap = split_ids[left] & split_ids[right]
            seed_overlap = split_seeds[left] & split_seeds[right]
            if id_overlap:
                total_failed += 1
                print(f"[FAIL] scenario_id overlap {left}/{right}: {len(id_overlap)}")
            if seed_overlap:
                total_failed += 1
                print(f"[FAIL] seed overlap {left}/{right}: {len(seed_overlap)}")

    if total_failed:
        raise SystemExit(f"Dataset audit failed: {total_failed} failure(s)")

    print("PASS: dataset audit completed with no detected errors.")


if __name__ == "__main__":
    main()
