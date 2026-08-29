from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        exclude_keys=EXCLUDE_KEYS,
    )


def build_model() -> TrajectoryConflictGNN:
    return TrajectoryConflictGNN(
        node_dim=MODEL_CONFIG.node_dim,
        edge_dim=MODEL_CONFIG.edge_dim,
        hidden_dim=MODEL_CONFIG.hidden_dim,
        heads=MODEL_CONFIG.heads,
        edge_mlp_dim=MODEL_CONFIG.edge_mlp_dim,
        dropout=MODEL_CONFIG.dropout,
    )


def compute_loss(model, batch):
    edge_logits, edge_probs, graph_count, graph_pair_count = model(batch, temperature=1.0)
    edge_labels = batch.edge_y.long().view(-1)

    edge_loss = F.cross_entropy(edge_logits, edge_labels)
    brier_loss = F.mse_loss(edge_probs, edge_labels.float())

    true_graph_count = batch.y.float().view(-1)
    pred_rate = graph_count / graph_pair_count
    true_rate = true_graph_count / graph_pair_count
    graph_rate_loss = F.smooth_l1_loss(pred_rate, true_rate)

    total_loss = (
        TRAIN_CONFIG.edge_loss_weight * edge_loss
        + TRAIN_CONFIG.brier_loss_weight * brier_loss
        + TRAIN_CONFIG.graph_rate_loss_weight * graph_rate_loss
    )
    return total_loss, edge_loss, brier_loss, graph_rate_loss


@torch.no_grad()
def evaluate(model, loader, device, temperature: float = 1.0) -> dict[str, Any]:
    model.eval()

    labels_all = []
    probs_all = []
    logits_all = []
    true_counts = []
    pred_counts = []

    loss_sum = 0.0
    graph_count = 0

    for batch in tqdm(loader, desc="validate", leave=False):
        batch = batch.to(device, non_blocking=True)
        edge_logits, edge_probs, predicted_count, _ = model(batch, temperature=temperature)

        labels = batch.edge_y.long().view(-1)
        loss_sum += float(F.cross_entropy(edge_logits / temperature, labels).item())
        graph_count += 1

        labels_all.append(labels.cpu())
        probs_all.append(edge_probs.cpu())
        logits_all.append(edge_logits.cpu())
        true_counts.append(batch.y.float().view(-1).cpu())
        pred_counts.append(predicted_count.cpu())

    labels_np = torch.cat(labels_all).numpy()
    probs_np = torch.cat(probs_all).numpy()
    logits = torch.cat(logits_all)
    true_count_np = torch.cat(true_counts).numpy()
    pred_count_np = torch.cat(pred_counts).numpy()

    metrics = binary_classification_metrics(labels_np, probs_np)
    metrics.update(graph_regression_metrics(true_count_np, pred_count_np))
    metrics["nll"] = loss_sum / max(graph_count, 1)

    return {
        "metrics": metrics,
        "labels": torch.from_numpy(labels_np),
        "probs": torch.from_numpy(probs_np),
        "logits": logits,
        "true_counts": torch.from_numpy(true_count_np),
        "pred_counts": torch.from_numpy(pred_count_np),
    }


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logits = logits.float().cpu()
    labels = labels.long().cpu()

    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=60,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.5, 5.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).clamp(0.5, 5.0).item())


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    best_pr_auc: float,
    calibrated_temperature: float | None = None,
    deployment_temperature: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "best_pr_auc": float(best_pr_auc),
            "dataset_version": TRAIN_CONFIG.dataset_version,
            "model_config": MODEL_CONFIG.__dict__,
            "train_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in TRAIN_CONFIG.__dict__.items()
            },
            "calibrated_temperature": calibrated_temperature,
            "deployment_temperature": deployment_temperature,
        },
        path,
    )


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(args) -> None:
    seed_everything(args.seed)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset_root = PROJECT_ROOT / "data" / "processed" / f"conflict_{args.dataset_version}"
    train_set = ShardedGraphDataset(dataset_root, "train", args.cache_size)
    val_set = ShardedGraphDataset(dataset_root, "val", args.cache_size)
    test_set = ShardedGraphDataset(dataset_root, "test", args.cache_size)

    train_loader = make_loader(train_set, args.batch_size, True, args.workers)
    val_loader = make_loader(val_set, args.batch_size, False, args.workers)
    test_loader = make_loader(test_set, args.batch_size, False, args.workers)

    model = build_model().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=TRAIN_CONFIG.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=TRAIN_CONFIG.lr_min,
    )

    amp_enabled = TRAIN_CONFIG.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    checkpoint_dir = TRAIN_CONFIG.checkpoint_dir
    result_dir = TRAIN_CONFIG.result_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoint_dir / "best_raw.pt"
    final_path = checkpoint_dir / "best_calibrated.pt"
    log_path = result_dir / "training_log.csv"
    metrics_path = result_dir / "final_metrics.json"

    if log_path.exists():
        log_path.unlink()

    best_pr_auc = -math.inf
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = np.zeros(4, dtype=np.float64)
        batches = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch:03d}/{args.epochs}")
        for batch in progress:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                total_loss, edge_loss, brier_loss, graph_loss = compute_loss(model, batch)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running += np.array([
                float(total_loss.detach()),
                float(edge_loss.detach()),
                float(brier_loss.detach()),
                float(graph_loss.detach()),
            ])
            batches += 1

            progress.set_postfix(loss=f"{running[0] / batches:.4f}")

        scheduler.step()

        val_result = evaluate(model, val_loader, device, temperature=1.0)
        val_metrics = val_result["metrics"]

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": running[0] / max(batches, 1),
            "train_edge_ce": running[1] / max(batches, 1),
            "train_brier": running[2] / max(batches, 1),
            "train_graph_rate": running[3] / max(batches, 1),
            "val_pr_auc": val_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_brier": val_metrics["brier"],
            "val_ece": val_metrics["ece"],
            "val_graph_mae": val_metrics["graph_mae"],
            "val_graph_rmse": val_metrics["graph_rmse"],
        }
        append_log(log_path, row)

        print(
            f"epoch={epoch:03d} "
            f"PR-AUC={val_metrics['pr_auc']:.4f} "
            f"F1={val_metrics['f1']:.4f} "
            f"Recall={val_metrics['recall']:.4f} "
            f"ECE={val_metrics['ece']:.4f} "
            f"GraphMAE={val_metrics['graph_mae']:.3f}"
        )

        if val_metrics["pr_auc"] > best_pr_auc + 1e-6:
            best_pr_auc = val_metrics["pr_auc"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, best_pr_auc)
        else:
            stale_epochs += 1

        if (
            epoch >= TRAIN_CONFIG.min_epochs
            and stale_epochs >= TRAIN_CONFIG.early_stop_patience
        ):
            print(f"early stop at epoch {epoch}; best epoch={best_epoch}")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    # Fit a single scalar temperature on validation logits.
    raw_val = evaluate(model, val_loader, device, temperature=1.0)
    calibrated_temperature = fit_temperature(raw_val["logits"], raw_val["labels"])
    deployment_temperature = max(
        calibrated_temperature,
        TRAIN_CONFIG.min_deployment_temperature,
    )

    model.set_temperature(deployment_temperature)

    calibrated_val = evaluate(
        model,
        val_loader,
        device,
        temperature=calibrated_temperature,
    )
    calibrated_test = evaluate(
        model,
        test_loader,
        device,
        temperature=calibrated_temperature,
    )
    deployment_test = evaluate(
        model,
        test_loader,
        device,
        temperature=deployment_temperature,
    )

    save_checkpoint(
        final_path,
        model,
        optimizer=None,
        epoch=best_epoch,
        best_pr_auc=best_pr_auc,
        calibrated_temperature=calibrated_temperature,
        deployment_temperature=deployment_temperature,
    )

    report = {
        "best_epoch": best_epoch,
        "best_raw_val_pr_auc": best_pr_auc,
        "calibrated_temperature": calibrated_temperature,
        "deployment_temperature": deployment_temperature,
        "validation_calibrated": calibrated_val["metrics"],
        "test_calibrated": calibrated_test["metrics"],
        "test_deployment_temperature": deployment_test["metrics"],
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"checkpoint: {final_path}")
    print(f"metrics:    {metrics_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train 2D Trajectory Conflict GNN")
    parser.add_argument("--dataset-version", default=TRAIN_CONFIG.dataset_version)
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG.batch_size)
    parser.add_argument("--workers", type=int, default=TRAIN_CONFIG.num_workers)
    parser.add_argument("--cache-size", type=int, default=TRAIN_CONFIG.shard_cache_size)
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG.learning_rate)
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG.seed)
    parser.add_argument("--device", type=str, default=None)
    return parser


if __name__ == "__main__":
    train(make_parser().parse_args())
