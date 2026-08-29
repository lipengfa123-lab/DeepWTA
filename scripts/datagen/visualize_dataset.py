from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
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


def load_graph(kind: str, split: str, index: int):
    dataset_dir = CONFIG.processed_dir / f"{kind}_{CONFIG.version}" / split
    current = 0
    for shard in sorted(dataset_dir.glob("part_*.pt")):
        graphs = torch.load(shard, map_location="cpu", weights_only=False)
        if current + len(graphs) > index:
            return graphs[index - current]
        current += len(graphs)
    raise IndexError(f"index {index} is outside split {split} with {current} graphs")


def load_background() -> np.ndarray:
    friendly = cv2.imread(str(CONFIG.maps.friendly_map), cv2.IMREAD_COLOR)
    enemy = cv2.imread(str(CONFIG.maps.enemy_map), cv2.IMREAD_COLOR)
    if friendly is None or enemy is None:
        raise FileNotFoundError("cannot load Friendly_Map.png / Enemy_Map.png")
    if friendly.shape != enemy.shape:
        raise ValueError("map image sizes do not match")
    merged = cv2.max(friendly, enemy)
    return cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)


def plot_wta_map(graph, save_path: Path | None) -> None:
    bg = load_background()
    base = graph.base_coords.numpy()
    target = graph.target_coords.numpy()
    allocation = graph.reference_allocation.numpy()

    active = np.argwhere(np.sum(allocation, axis=0) > 0)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(bg)

    for base_id, target_id in active:
        ax.plot(
            [base[base_id, 1], target[target_id, 1]],
            [base[base_id, 0], target[target_id, 0]],
            linewidth=0.9,
            alpha=0.55,
        )

    ax.scatter(base[:, 1], base[:, 0], s=24, marker="s", label="Bases", zorder=3)
    ax.scatter(target[:, 1], target[:, 0], s=28, marker="^", label="Targets", zorder=3)
    ax.set_title(
        f"WTA scenario: {graph.scenario_id} | "
        f"N={int(graph.num_bases)}, M={int(graph.num_targets)}, "
        f"active routes={len(active)}"
    )
    ax.legend()
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"saved: {save_path}")
    else:
        plt.show()


def plot_wta_schemes(graph, target_id: int, save_path: Path | None) -> None:
    resource = graph.scheme_resource.numpy()[:, target_id]
    levels = graph.G.numpy()[:, target_id].astype(int)
    threshold = int(graph.G_th.numpy()[target_id])

    order = np.argsort(levels * 100000 + resource)
    labels = [f"S{i}\nG={levels[i]}" for i in order]
    values = resource[order]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values)
    ax.set_ylabel("Expected resource")
    ax.set_title(
        f"Target {target_id}: scheme resources | G_th={threshold} | "
        f"scenario={graph.scenario_id}"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"saved: {save_path}")
    else:
        plt.show()


def plot_conflict(graph, save_path: Path | None) -> None:
    bg = load_background()
    base = graph.base_coords.numpy()
    target = graph.target_coords.numpy()
    routes = graph.routes.numpy().astype(int)
    pairs = graph.undirected_pairs.numpy().astype(int)
    labels = graph.undirected_edge_y.numpy().astype(int)

    positive_route_ids = set()
    for pair, label in zip(pairs, labels):
        if label == 1:
            positive_route_ids.update(pair.tolist())

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(bg)

    for route_id, (base_id, target_id) in enumerate(routes):
        is_cross = route_id in positive_route_ids
        ax.plot(
            [base[base_id, 1], target[target_id, 1]],
            [base[base_id, 0], target[target_id, 0]],
            linewidth=1.4 if is_cross else 0.7,
            alpha=0.8 if is_cross else 0.25,
        )

    ax.scatter(base[:, 1], base[:, 0], s=20, marker="s", label="Bases", zorder=3)
    ax.scatter(target[:, 1], target[:, 0], s=24, marker="^", label="Targets", zorder=3)

    cross_count = int(np.sum(labels))
    before = int(graph.total_pairs_before_pruning)
    after = int(graph.candidate_pairs_after_aabb)
    ax.set_title(
        f"2D route-conflict graph: {graph.scenario_id} | "
        f"routes={len(routes)}, cross pairs={cross_count}, "
        f"pairs {before}->{after} after deterministic pruning"
    )
    ax.legend()
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"saved: {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize generated DeepWTA datasets")
    parser.add_argument("kind", choices=["wta", "conflict"])
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--target", type=int, default=0, help="target index for --view schemes")
    parser.add_argument("--view", choices=["map", "schemes"], default="map")
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    graph = load_graph(args.kind, args.split, args.index)

    if args.kind == "wta":
        if args.view == "schemes":
            if args.target < 0 or args.target >= int(graph.num_targets):
                raise ValueError("target index is outside this scenario")
            plot_wta_schemes(graph, args.target, args.save)
        else:
            plot_wta_map(graph, args.save)
    else:
        if args.view != "map":
            raise ValueError("conflict dataset supports --view map only")
        plot_conflict(graph, args.save)


if __name__ == "__main__":
    main()
