from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from deepwta.utils.env import add_project_paths  # noqa: E402
PROJECT_ROOT = add_project_paths()

from configs.conflict_train_config import MODEL_CONFIG, TRAIN_CONFIG
from deepwta.data.sharded_dataset import ShardedGraphDataset
from deepwta.models.trajectory_conflict_gnn import TrajectoryConflictGNN
from deepwta.utils.conflict_metrics import (
    binary_classification_metrics,
    graph_regression_metrics,
)


EXCLUDE_KEYS = [
    "routes",
    "base_coords",
    "target_coords",
    "undirected_pairs",
    "undirected_edge_y",
    "total_pairs_before_pruning",
    "candidate_pairs_after_aabb",
    "pairs_after_sampling",
    "scenario_seed",
    "scenario_id",
]


def build_model(model_cfg: dict) -> TrajectoryConflictGNN:
    return TrajectoryConflictGNN(
        node_dim=model_cfg["node_dim"],
        edge_dim=model_cfg["edge_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        heads=model_cfg["heads"],
        edge_mlp_dim=model_cfg["edge_mlp_dim"],
        dropout=model_cfg["dropout"],
    )


@torch.no_grad()
def run_eval(model, loader, device, temperature: float):
    model.eval()
    labels = []
    probs = []
    true_counts = []
    pred_counts = []

    for batch in tqdm(loader, desc="test"):
        batch = batch.to(device, non_blocking=True)
        _, edge_probs, graph_count, _ = model(batch, temperature=temperature)
        labels.append(batch.edge_y.long().view(-1).cpu())
        probs.append(edge_probs.cpu())
        true_counts.append(batch.y.float().view(-1).cpu())
        pred_counts.append(graph_count.cpu())

    labels = torch.cat(labels).numpy()
    probs = torch.cat(probs).numpy()
    true_counts = torch.cat(true_counts).numpy()
    pred_counts = torch.cat(pred_counts).numpy()

    metrics = binary_classification_metrics(labels, probs)
    metrics.update(graph_regression_metrics(true_counts, pred_counts))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Trajectory Conflict GNN")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=TRAIN_CONFIG.checkpoint_dir / "best_calibrated.pt",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--temperature",
        choices=["raw", "calibrated", "deployment"],
        default="calibrated",
    )
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG.batch_size)
    parser.add_argument("--workers", type=int, default=TRAIN_CONFIG.num_workers)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = checkpoint["model_config"]
    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])

    dataset_version = checkpoint["dataset_version"]
    dataset_root = PROJECT_ROOT / "data" / "processed" / f"conflict_{dataset_version}"
    dataset = ShardedGraphDataset(
        dataset_root,
        args.split,
        TRAIN_CONFIG.shard_cache_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        exclude_keys=EXCLUDE_KEYS,
    )

    if args.temperature == "raw":
        temperature = 1.0
    elif args.temperature == "deployment":
        temperature = float(checkpoint["deployment_temperature"])
    else:
        temperature = float(checkpoint["calibrated_temperature"])

    metrics = run_eval(model, loader, device, temperature)
    report = {
        "split": args.split,
        "temperature_mode": args.temperature,
        "temperature": temperature,
        **metrics,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
