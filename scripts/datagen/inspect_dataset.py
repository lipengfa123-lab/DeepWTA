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


def iter_graphs(dataset_dir: Path, split: str, limit: int):
    seen = 0
    for shard in sorted((dataset_dir / split).glob("part_*.pt")):
        graphs = torch.load(shard, map_location="cpu", weights_only=False)
        for graph in graphs:
            yield graph
            seen += 1
            if seen >= limit:
                return


def inspect_wta(dataset_dir: Path, split: str, limit: int) -> None:
    reachability = []
    node_dims = []
    edge_dims = []
    sizes = []
    size_ratios = []
    hierarchy_ok = 0
    total = 0

    for graph in iter_graphs(dataset_dir, split, limit):
        total += 1
        reachability.append(float(graph.X.float().mean()))
        node_dims.append(int(graph.x.shape[1]))
        edge_dims.append(int(graph.edge_attr.shape[1]))

        n = int(graph.num_bases)
        m = int(graph.num_targets)
        sizes.append((n, m))
        size_ratios.append(n / m)

        resource = graph.scheme_resource.numpy()
        g = graph.G.numpy()
        good = True
        for target in range(m):
            means = [
                float(np.mean(resource[g[:, target] == level, target]))
                for level in (1, 2, 3)
            ]
            good &= means[1] >= means[0] * (1.0 + CONFIG.wta.level_resource_gap) - 1e-5
            good &= means[2] >= means[1] * (1.0 + CONFIG.wta.level_resource_gap) - 1e-5
        hierarchy_ok += int(good)

    print(f"graphs: {total}")
    print(f"mean X reachable rate: {np.mean(reachability):.4f}")
    print(f"node_dim: {sorted(set(node_dims))}")
    print(f"edge_dim: {sorted(set(edge_dims))}")
    print(f"resource hierarchy pass: {hierarchy_ok}/{total}")
    print(
        f"N/M ratio range: [{np.min(size_ratios):.3f}, {np.max(size_ratios):.3f}] "
        f"(required [{CONFIG.wta.min_size_ratio:.3f}, {CONFIG.wta.max_size_ratio:.3f}])"
    )
    print(f"sample sizes: {sizes[:12]}")


def inspect_conflict(dataset_dir: Path, split: str, limit: int) -> None:
    cross_rates = []
    pruning_rates = []
    node_dims = []
    edge_dims = []
    size_ratios = []
    total = 0

    for graph in iter_graphs(dataset_dir, split, limit):
        total += 1
        labels = graph.undirected_edge_y.numpy()
        cross_rates.append(float(np.mean(labels)))

        before = max(int(graph.total_pairs_before_pruning), 1)
        after = int(graph.candidate_pairs_after_aabb)
        pruning_rates.append(1.0 - after / before)

        node_dims.append(int(graph.x.shape[1]))
        edge_dims.append(int(graph.edge_attr.shape[1]))

        n = int(graph.base_coords.shape[0])
        m = int(graph.target_coords.shape[0])
        size_ratios.append(n / m)

    print(f"graphs: {total}")
    print(f"mean Cross ratio after sampling: {np.mean(cross_rates):.4f}")
    print(f"mean deterministic pair-pruning rate: {np.mean(pruning_rates):.4f}")
    print(f"node_dim: {sorted(set(node_dims))}")
    print(f"edge_dim: {sorted(set(edge_dims))}")
    print(
        f"N/M ratio range: [{np.min(size_ratios):.3f}, {np.max(size_ratios):.3f}] "
        f"(required [{CONFIG.conflict.min_size_ratio:.3f}, {CONFIG.conflict.max_size_ratio:.3f}])"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["wta", "conflict"])
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    dataset_dir = CONFIG.processed_dir / f"{args.kind}_{CONFIG.version}"
    if args.kind == "wta":
        inspect_wta(dataset_dir, args.split, args.limit)
    else:
        inspect_conflict(dataset_dir, args.split, args.limit)


if __name__ == "__main__":
    main()
