from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.dataset_config import CONFIG as DATA_CONFIG
from configs.deepwta_model_config import MODEL_CONFIG
from configs.deepwta_train_config import TRAIN_CONFIG, PhaseWeights
from configs.deepwta_t1_config import T1_CONFIG
from deepwta.data.map_data import load_map_context
from deepwta.data.sharded_dataset import ShardedGraphDataset
from deepwta.mechanisms.conflict_regularizer import (
    FrozenConflictRegularizer,
    load_frozen_conflict_gnn,
)
from deepwta.mechanisms.deepwta_loss_t1 import DeepWTALossT1
from deepwta.training.loss_normalizer import EMALossNormalizer
from deepwta.training.robust_constraint_controller import (
    RobustConstraintControllerV2,
)
from configs.deepwta_t2b_profiles import (
    TACTICAL_BUDGET,
    TACTICAL_PROFILES,
    TacticalProfile,
    get_tactical_profile,
)
from deepwta.models.deepwta_model import DeepWTAModel


EXCLUDE_KEYS = [
    # Variable-shape physical tensors are already encoded into x/edge_attr and
    # must not be naively concatenated by PyG.
    "u",
    "B",
    "Q",
    "U",
    "G",
    "G_th",
    "X",
    "D",
    "base_coords",
    "target_coords",
    "reference_scheme",
    "reference_allocation",
    "scheme_resource",
    "num_bases",
    "num_targets",
    "num_weapons",
    "num_schemes",
    "scenario_seed",
    "scenario_id",
]


class EpochRandomSampler(Sampler[int]):
    """Deterministic epoch-wise random subset without replacement."""

    def __init__(
        self,
        dataset_size: int,
        samples_per_epoch: int,
        seed: int,
    ):
        self.dataset_size = int(dataset_size)
        self.samples_per_epoch = min(
            max(int(samples_per_epoch), 1),
            self.dataset_size,
        )
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1_000_003)
        order = torch.randperm(
            self.dataset_size,
            generator=generator,
        )
        return iter(order[: self.samples_per_epoch].tolist())

    def __len__(self) -> int:
        return self.samples_per_epoch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: to_jsonable(val)
            for key, val in asdict(value).items()
        }
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def build_model(
    use_scheme_efficiency_features: bool = False,
    tactical_assignment_bias: str = "none",
    tactical_bias_alpha_init: float = 0.05,
    assignment_temperature: float = 1.0,
    zero_inventory_assignment_mask: bool = False,
    resource_feasible_scheme_mask: bool = False,
    positive_inventory_contention_lambda: float = 0.0,
    redistribution_mode: str = "none",
    redistribution_beta: float = 0.0,
    redistribution_iterations: int = 2,
    redistribution_eta: float = 0.75,
) -> DeepWTAModel:
    c = MODEL_CONFIG
    return DeepWTAModel(
        node_dim=c.node_dim,
        edge_dim=c.edge_dim,
        num_weapons=c.num_weapons,
        num_schemes=c.num_schemes,
        inventory_max=c.inventory_max,
        demand_max=c.demand_max,
        hidden_dim=c.hidden_dim,
        heads=c.heads,
        num_encoder_layers=c.num_encoder_layers,
        scheme_dim=c.scheme_dim,
        scheme_hidden_dim=c.scheme_hidden_dim,
        edge_mlp_dim=c.edge_mlp_dim,
        dropout=c.dropout,
        survival_floor=c.survival_floor,
        use_inventory_projection=c.use_inventory_projection,
        use_scheme_efficiency_features=use_scheme_efficiency_features,
        tactical_assignment_bias=tactical_assignment_bias,
        tactical_bias_alpha_init=tactical_bias_alpha_init,
        assignment_temperature=assignment_temperature,
        zero_inventory_assignment_mask=zero_inventory_assignment_mask,
        resource_feasible_scheme_mask=resource_feasible_scheme_mask,
        positive_inventory_contention_lambda=positive_inventory_contention_lambda,
        redistribution_mode=redistribution_mode,
        redistribution_beta=redistribution_beta,
        redistribution_iterations=redistribution_iterations,
        redistribution_eta=redistribution_eta,
    )


def initialize_model_weights(model, checkpoint: dict) -> None:
    """Load B0 weights while allowing only explicitly enabled ablation parameters."""
    incompatible = model.load_state_dict(
        checkpoint["model_state"], strict=False
    )
    expected_missing: set[str] = set()
    if model.use_scheme_efficiency_features:
        expected_missing.update(
            key for key in model.state_dict()
            if key.startswith("scheme_efficiency_encoder.")
        )
    if model.tactical_assignment_bias in {"distance", "both"}:
        expected_missing.add("tactical_distance_raw_alpha")
    if model.tactical_assignment_bias in {"interception", "both"}:
        expected_missing.add("tactical_interception_raw_alpha")
    unexpected = list(incompatible.unexpected_keys)
    actual_missing = set(incompatible.missing_keys)
    if unexpected or actual_missing != expected_missing:
        raise RuntimeError(
            "incompatible initialization checkpoint: "
            f"missing={sorted(actual_missing)}, "
            f"expected_missing={sorted(expected_missing)}, "
            f"unexpected={unexpected}"
        )


def make_loader(
    dataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    sampler=None,
    shuffle: bool = False,
):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        exclude_keys=EXCLUDE_KEYS,
    )
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def make_limited_loader(
    dataset,
    limit: int,
    batch_size: int,
    workers: int,
    device: torch.device,
):
    if limit <= 0 or limit >= len(dataset):
        subset = dataset
    else:
        subset = torch.utils.data.Subset(dataset, range(limit))
    return make_loader(
        subset,
        batch_size=batch_size,
        workers=workers,
        device=device,
        shuffle=False,
    )


def cosine_warmup_factor(
    epoch_zero_based: int,
    total_epochs: int,
    warmup_epochs: int,
    min_ratio: float,
) -> float:
    if epoch_zero_based < warmup_epochs:
        return max(
            (epoch_zero_based + 1) / max(warmup_epochs, 1),
            min_ratio,
        )

    remaining = max(total_epochs - warmup_epochs, 1)
    progress = min(
        max(
            (epoch_zero_based - warmup_epochs) / remaining,
            0.0,
        ),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def scheme_tau(epoch: int, total_epochs: int) -> float:
    start = TRAIN_CONFIG.scheme_tau_start
    end = TRAIN_CONFIG.scheme_tau_end
    if total_epochs <= 1:
        return end
    progress = (epoch - 1) / (total_epochs - 1)
    return start * ((end / start) ** progress)


def phase_weights(phase: str) -> PhaseWeights:
    if phase == "FEASIBILITY":
        return TRAIN_CONFIG.feasibility_weights
    if phase == "BALANCED":
        return TRAIN_CONFIG.balanced_weights
    if phase == "FULL_OPT":
        return TRAIN_CONFIG.full_weights
    raise ValueError(f"unknown phase: {phase}")


def objective_scale(
    phase: str,
    epoch: int,
    phase_start_epoch: int,
) -> float:
    if phase == "FEASIBILITY":
        return 0.0
    if phase == "FULL_OPT":
        return 1.0
    elapsed = epoch - phase_start_epoch + 1
    return min(
        1.0,
        elapsed / max(TRAIN_CONFIG.objective_ramp_epochs, 1),
    )


def ema_training_policy(
    requested_mode: str,
    phase: str,
    epoch: int,
    phase_start_epoch: int,
    ramp_epochs: int,
) -> tuple[str, float, bool]:
    """Return (criterion_mode, EMA strength, update_ema).

    FEASIBILITY: original loss, no EMA updates.
    BALANCED   : original loss, collect EMA scales only.
    FULL_OPT   : gradually blend raw -> EMA-normalized tactical losses.
    """
    if requested_mode == "raw":
        return "raw", 0.0, False

    if phase == "FEASIBILITY":
        return "raw", 0.0, False

    if phase == "BALANCED":
        return "raw", 0.0, True

    elapsed = epoch - phase_start_epoch + 1
    strength = min(1.0, elapsed / max(int(ramp_epochs), 1))
    return "ema", strength, True


def maybe_promote_phase(
    phase: str,
    epoch: int,
    phase_start_epoch: int,
    ema_shortage: float,
) -> tuple[str, int]:
    if phase == "FEASIBILITY":
        enough_epochs = epoch >= TRAIN_CONFIG.feasibility_min_epochs
        good_enough = ema_shortage <= TRAIN_CONFIG.feasibility_shortage_threshold
        forced = epoch >= TRAIN_CONFIG.feasibility_max_epochs
        if enough_epochs and (good_enough or forced):
            return "BALANCED", epoch + 1

    if phase == "BALANCED":
        balanced_age = epoch - phase_start_epoch + 1
        enough_epochs = balanced_age >= TRAIN_CONFIG.balanced_min_epochs
        good_enough = ema_shortage <= TRAIN_CONFIG.balanced_shortage_threshold
        forced = epoch >= TRAIN_CONFIG.balanced_max_end_epoch
        if enough_epochs and (good_enough or forced):
            return "FULL_OPT", epoch + 1

    return phase, phase_start_epoch


def scheme_mode_for_phase(phase: str) -> tuple[str, bool]:
    if phase == "FULL_OPT":
        return "gumbel", TRAIN_CONFIG.full_phase_gumbel_hard
    return "soft", False


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row.keys()),
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def scalar_stats(stats: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().cpu())
        for key, value in stats.items()
    }


def accumulate(
    running: dict[str, float],
    stats: dict[str, float],
    weight: int = 1,
) -> None:
    for key, value in stats.items():
        running[key] = running.get(key, 0.0) + value * weight
    running["_weight"] = running.get("_weight", 0.0) + weight


def average_running(running: dict[str, float]) -> dict[str, float]:
    denominator = max(running.get("_weight", 0.0), 1.0)
    return {
        key: value / denominator
        for key, value in running.items()
        if key != "_weight"
    }



def runtime_phase_weights(
    phase: str,
    constraint_controller: RobustConstraintControllerV2,
    tactical_profile: TacticalProfile,
) -> dict[str, float]:
    """Weights used in the next optimization epoch.

    FEASIBILITY/BALANCED stay identical to the existing curriculum.

    FULL_OPT:
      - demand/inventory multipliers come from the robust controller;
      - tactical semantic proportions come from the chosen style;
      - the controller soft gate scales ALL tactical soft objectives.

    The last point is a deliberate T2b fix: in T2a dual_fixed the gate was
    printed but was only applied inside DWA variants, so it had no effect on
    dual_fixed training.
    """
    base = phase_weights(phase).as_dict()

    if phase != "FULL_OPT":
        return base

    base["demand"] = constraint_controller.lambda_demand
    base["inventory_pre"] = constraint_controller.lambda_inventory

    gate = constraint_controller.soft_gate()
    profile = tactical_profile.effective_dict()
    base["ammo"] = gate * profile["ammo"]
    base["crossing"] = gate * profile["crossing"]
    base["distance"] = gate * profile["distance"]
    base["interception"] = gate * profile["interception"]
    return base


def full_selection_score(
    metrics: dict[str, float],
    tactical_profile: TacticalProfile,
) -> float:
    """Ungated raw tactical score among constraint-feasible FULL epochs."""
    weights = tactical_profile.effective_dict()
    return (
        TRAIN_CONFIG.full_weights.scheme_entropy * metrics["scheme_entropy"]
        + weights["ammo"] * metrics["ammo"]
        + weights["crossing"] * metrics["crossing"]
        + weights["distance"] * metrics["distance"]
        + weights["interception"] * metrics["interception"]
    )


def train_one_epoch(
    model,
    criterion,
    loader,
    sampler,
    optimizer,
    scaler,
    device,
    epoch: int,
    total_epochs: int,
    phase: str,
    phase_start_epoch: int,
    amp_enabled: bool,
    requested_loss_mode: str,
    ema_ramp_epochs: int,
    runtime_weights: dict[str, float],
) -> dict[str, float]:
    model.train()
    criterion.train()
    sampler.set_epoch(epoch)

    weights = dict(runtime_weights)
    obj_scale = objective_scale(
        phase,
        epoch,
        phase_start_epoch,
    )
    tau = scheme_tau(epoch, total_epochs)
    mode, gumbel_hard = scheme_mode_for_phase(phase)
    criterion_mode, ema_strength, update_ema = ema_training_policy(
        requested_loss_mode,
        phase,
        epoch,
        phase_start_epoch,
        ema_ramp_epochs,
    )

    optimizer.zero_grad(set_to_none=True)
    running: dict[str, float] = {}
    accumulation_counter = 0

    progress = tqdm(
        loader,
        desc=f"{phase} epoch {epoch:03d}/{total_epochs}",
    )

    for batch_index, batch in enumerate(progress, start=1):
        batch = batch.to(device, non_blocking=True)
        batch_graphs = int(batch.num_graphs)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model(
                batch,
                scheme_mode=mode,
                scheme_tau=tau,
                gumbel_hard=gumbel_hard,
            )
            loss, stats = criterion(
                batch,
                output,
                weights=weights,
                objective_scale=obj_scale,
                conflict_sampling_step=epoch,
                loss_mode=criterion_mode,
                ema_strength=ema_strength,
                update_ema=update_ema,
            )
            scaled_loss = loss / TRAIN_CONFIG.grad_accum_steps

        scaler.scale(scaled_loss).backward()
        accumulation_counter += 1

        if (
            accumulation_counter >= TRAIN_CONFIG.grad_accum_steps
            or batch_index == len(loader)
        ):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                TRAIN_CONFIG.grad_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0

        stats_float = scalar_stats(stats)
        accumulate(running, stats_float, batch_graphs)
        avg = average_running(running)

        progress.set_postfix(
            loss=f"{avg['loss']:.4f}",
            dem=f"{avg['demand']:.3f}",
            cross=f"{avg['crossing']:.3f}",
            sat=f"{avg['demand_satisfaction_rate']:.3f}",
            tau=f"{tau:.2f}",
            ema=f"{ema_strength:.2f}",
        )

    result = average_running(running)
    result["scheme_tau"] = tau
    result["objective_scale"] = obj_scale
    result["ema_strength"] = ema_strength
    result["criterion_loss_mode"] = 1.0 if criterion_mode == "ema" else 0.0
    return result


@torch.no_grad()
def evaluate(
    model,
    criterion,
    loader,
    device,
    split: str,
) -> dict[str, float]:
    model.eval()
    criterion.eval()
    running: dict[str, float] = {}
    weights = TRAIN_CONFIG.full_weights.as_dict()

    progress = tqdm(loader, desc=f"eval:{split}")
    for batch in progress:
        batch = batch.to(device, non_blocking=True)
        batch_graphs = int(batch.num_graphs)

        output = model(
            batch,
            scheme_mode="hard",
            scheme_tau=TRAIN_CONFIG.scheme_tau_end,
            gumbel_hard=False,
        )
        _, stats = criterion(
            batch,
            output,
            weights=weights,
            objective_scale=1.0,
            # Model selection remains on the original fixed raw objective, so
            # epoch-to-epoch validation scores are comparable even while EMA
            # scales evolve during training.
            conflict_sampling_step=0,
            loss_mode="raw",
            ema_strength=0.0,
            update_ema=False,
        )
        accumulate(
            running,
            scalar_stats(stats),
            batch_graphs,
        )

    return average_running(running)


def validation_score(metrics: dict[str, float]) -> float:
    # `loss` already contains a strong demand term. The second term makes model
    # selection explicitly prefer hard-scheme feasibility when total losses tie.
    return (
        metrics["loss"]
        + TRAIN_CONFIG.validation_infeasibility_weight
        * (1.0 - metrics["demand_satisfaction_rate"])
    )


def save_checkpoint(
    path: Path,
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    phase: str,
    phase_start_epoch: int,
    ema_shortage: float,
    best_score: float,
    best_epoch: int,
    stale_epochs: int,
    run_name: str,
    seed: int,
    loss_mode: str,
    ema_beta: float,
    ema_ramp_epochs: int,
    tactical_style: str,
    tactical_profile: TacticalProfile,
    constraint_controller: RobustConstraintControllerV2,
    best_full_score: float,
    best_full_epoch: int,
    full_stale_epochs: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = to_jsonable(MODEL_CONFIG)
    model_config["use_scheme_efficiency_features"] = bool(
        model.use_scheme_efficiency_features
    )
    model_config["tactical_assignment_bias"] = str(
        model.tactical_assignment_bias
    )
    model_config["tactical_bias_alpha_init"] = float(
        model.tactical_bias_alpha_init
    )
    model_config["assignment_temperature"] = float(
        model.assignment_temperature
    )
    model_config["redistribution_mode"] = str(model.redistribution_mode)
    model_config["redistribution_beta"] = float(model.redistribution_beta)
    model_config["redistribution_iterations"] = int(
        model.redistribution_iterations
    )
    model_config["redistribution_eta"] = float(model.redistribution_eta)
    torch.save(
        {
            "model_state": model.state_dict(),
            "normalizer_state": criterion.normalizer.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": int(epoch),
            "phase": str(phase),
            "phase_start_epoch": int(phase_start_epoch),
            "ema_shortage": float(ema_shortage),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "stale_epochs": int(stale_epochs),
            "dataset_version": TRAIN_CONFIG.dataset_version,
            "run_name": str(run_name),
            "seed": int(seed),
            "loss_mode": str(loss_mode),
            "ema_beta": float(ema_beta),
            "ema_ramp_epochs": int(ema_ramp_epochs),
            "training_variant": "T2B_ROBUST_BALANCED",
            "tactical_style": str(tactical_style),
            "tactical_profile": tactical_profile.as_dict(),
            "tactical_budget": float(TACTICAL_BUDGET),
            "tactical_effective_weights": tactical_profile.effective_dict(),
            "tactical_profile_description": tactical_profile.description,
            "constraint_controller_state": constraint_controller.state_dict(),
            "best_full_score": float(best_full_score),
            "best_full_epoch": int(best_full_epoch),
            "full_stale_epochs": int(full_stale_epochs),
            "model_config": model_config,
            "train_config": to_jsonable(TRAIN_CONFIG),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    constraint_controller=None,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    if checkpoint.get("normalizer_state"):
        criterion.normalizer.load_state_dict(checkpoint["normalizer_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])
    if (
        constraint_controller is not None
        and checkpoint.get("constraint_controller_state")
    ):
        constraint_controller.load_state_dict(
            checkpoint["constraint_controller_state"]
        )
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train DeepWTA T2b robust balanced model"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=TRAIN_CONFIG.dataset_dir,
        help=(
            "Sharded dataset root containing train/val/test. The default "
            "preserves the original T2b training contract."
        ),
    )
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG.batch_size)
    parser.add_argument("--workers", type=int, default=TRAIN_CONFIG.num_workers)
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG.learning_rate)
    parser.add_argument(
        "--train-graphs",
        type=int,
        default=TRAIN_CONFIG.train_graphs_per_epoch,
    )
    parser.add_argument(
        "--run-name",
        default="t2b_balanced_seed2026",
        help="Independent checkpoint/result namespace.",
    )
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG.seed)
    parser.add_argument(
        "--loss-mode",
        choices=("raw", "ema"),
        default="ema",
        help="raw=T0-compatible objective; ema=T1 training objective.",
    )
    parser.add_argument("--ema-beta", type=float, default=T1_CONFIG.ema_beta)
    parser.add_argument(
        "--ema-ramp-epochs",
        type=int,
        default=T1_CONFIG.full_normalization_ramp_epochs,
    )
    parser.add_argument(
        "--tactical-style",
        choices=tuple(sorted(TACTICAL_PROFILES)),
        default="balanced",
        help=(
            "Semantic tactical preference. 'balanced' is the recommended "
            "all profile shares sum to 1.0; the frozen T2b tactical budget remains 0.77."
        ),
    )
    parser.add_argument(
        "--constraint-target-mode",
        choices=("auto", "fixed"),
        default="auto",
    )
    parser.add_argument("--balanced-window", type=int, default=10)
    parser.add_argument("--robust-window", type=int, default=3)
    parser.add_argument("--controller-update-interval", type=int, default=2)
    parser.add_argument("--controller-warmup-epochs", type=int, default=3)
    parser.add_argument("--hysteresis-low", type=float, default=0.90)
    parser.add_argument("--hysteresis-high", type=float, default=1.10)
    parser.add_argument("--dual-eta-up", type=float, default=0.10)
    parser.add_argument("--dual-eta-down", type=float, default=0.035)
    parser.add_argument("--soft-gate-floor", type=float, default=0.45)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=None,
        help=(
            "Weights-only warm start for an incremental architecture. Restores "
            "normalizer/controller state but creates a fresh optimizer."
        ),
    )
    parser.add_argument(
        "--use-scheme-efficiency-features",
        action="store_true",
        help="Enable the Phase-1 scheme-efficiency residual branch.",
    )
    parser.add_argument(
        "--tactical-assignment-bias",
        choices=("none", "distance", "interception", "both"),
        default="none",
        help="Phase-2 explicit route-logit prior; default keeps the B0 baseline.",
    )
    parser.add_argument(
        "--tactical-bias-alpha-init",
        type=float,
        default=0.05,
        help="Initial effective softplus alpha for the Phase-2 route prior.",
    )
    parser.add_argument(
        "--assignment-temperature",
        type=float,
        default=1.0,
        help="Phase-3A base-assignment softmax temperature in (0,1].",
    )
    parser.add_argument(
        "--redistribution-mode",
        choices=("none", "capacity", "network"),
        default="none",
    )
    parser.add_argument("--redistribution-beta", type=float, default=0.0)
    parser.add_argument("--redistribution-iterations", type=int, default=2)
    parser.add_argument("--redistribution-eta", type=float, default=0.75)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Do not load or evaluate the test split during model development.",
    )
    parser.add_argument(
        "--no-conflict",
        action="store_true",
        help="Ablation/debug: disable frozen trajectory-conflict regularization.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny end-to-end training smoke test.",
    )
    parser.add_argument(
        "--smoke-full",
        action="store_true",
        help=(
            "Smoke-test the actual EMA/FULL path. Implies --smoke and starts "
            "directly in FULL_OPT; use only for integration testing."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_full:
        args.smoke = True
        if args.constraint_target_mode == "auto":
            args.constraint_target_mode = "fixed"
    if args.resume is not None and args.initialize_from is not None:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    if args.use_scheme_efficiency_features and args.tactical_assignment_bias != "none":
        raise ValueError("Phase-2 tactical bias must use the B0 scheme decoder")
    if not 0.0 < args.assignment_temperature <= 1.0:
        raise ValueError("--assignment-temperature must be in (0,1]")
    if args.assignment_temperature != 1.0 and (
        args.use_scheme_efficiency_features
        or args.tactical_assignment_bias != "none"
    ):
        raise ValueError("Phase-3A temperature must use the formal B0 model")
    if args.redistribution_mode != "none" and (
        args.use_scheme_efficiency_features
        or args.tactical_assignment_bias != "none"
        or args.assignment_temperature != 1.0
    ):
        raise ValueError("Phase-3B redistribution must use the formal B0 model")
    if not 0.0 <= args.ema_beta < 1.0:
        raise ValueError("--ema-beta must satisfy 0 <= beta < 1")
    if args.ema_ramp_epochs <= 0:
        raise ValueError("--ema-ramp-epochs must be positive")
    if args.dual_eta_up <= 0.0:
        raise ValueError("--dual-eta-up must be positive")
    if args.dual_eta_down <= 0.0:
        raise ValueError("--dual-eta-down must be positive")
    if not args.hysteresis_low < 1.0 < args.hysteresis_high:
        raise ValueError("hysteresis must satisfy low < 1 < high")
    if not 0.0 < args.soft_gate_floor <= 1.0:
        raise ValueError("--soft-gate-floor must be in (0,1]")
    seed_everything(args.seed)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp_enabled = TRAIN_CONFIG.use_amp and device.type == "cuda"
    dataset_dir = args.dataset_dir.expanduser().resolve()

    if args.smoke:
        args.epochs = 2
        args.train_graphs = min(24, args.train_graphs)
        args.batch_size = min(2, args.batch_size)
        val_limit = 12
        test_limit = 12
        print("SMOKE MODE: epochs=2, train<=24, val/test<=12")
    else:
        val_limit = TRAIN_CONFIG.val_graph_limit
        test_limit = TRAIN_CONFIG.test_graph_limit

    print(f"device          : {device}")
    print(f"dataset         : {dataset_dir}")
    print(f"epochs          : {args.epochs}")
    print(f"batch size      : {args.batch_size}")
    print(f"grad accumulation: {TRAIN_CONFIG.grad_accum_steps}")
    print(f"graphs / epoch  : {args.train_graphs}")
    print(f"run name        : {args.run_name}")
    print(f"seed            : {args.seed}")
    print(f"loss mode       : {args.loss_mode}")
    print(f"ema beta        : {args.ema_beta}")
    print(f"ema FULL ramp   : {args.ema_ramp_epochs} epochs")
    tactical_profile = get_tactical_profile(args.tactical_style)
    print("training variant: T2B_ROBUST_BALANCED")
    print(f"tactical style  : {args.tactical_style}")
    print(f"style shares    : {tactical_profile.as_dict()} (sum=1.0)")
    print(f"tactical budget : {TACTICAL_BUDGET:.2f} (frozen T2b scale)")
    print(f"base eff weights: {tactical_profile.effective_dict()}")
    print(f"style meaning   : {tactical_profile.description}")
    print(f"constraint target: {args.constraint_target_mode}")
    print(
        "controller      : "
        f"robust_window={args.robust_window}, "
        f"update_interval={args.controller_update_interval}, "
        f"hysteresis=[{args.hysteresis_low},{args.hysteresis_high}]"
    )
    print(
        "dual eta        : "
        f"up={args.dual_eta_up}, down={args.dual_eta_down}"
    )
    print(f"soft gate floor : {args.soft_gate_floor}")

    train_set = ShardedGraphDataset(
        dataset_dir,
        "train",
        TRAIN_CONFIG.shard_cache_size,
    )
    val_set = ShardedGraphDataset(
        dataset_dir,
        "val",
        TRAIN_CONFIG.shard_cache_size,
    )
    test_set = None
    if not args.validation_only:
        test_set = ShardedGraphDataset(
            dataset_dir,
            "test",
            TRAIN_CONFIG.shard_cache_size,
        )

    train_samples = (
        len(train_set)
        if args.train_graphs <= 0
        else min(args.train_graphs, len(train_set))
    )
    train_sampler = EpochRandomSampler(
        dataset_size=len(train_set),
        samples_per_epoch=train_samples,
        seed=args.seed,
    )

    train_loader = make_loader(
        train_set,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        sampler=train_sampler,
    )
    val_loader = make_limited_loader(
        val_set,
        limit=val_limit,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    test_loader = None
    if test_set is not None:
        test_loader = make_limited_loader(
            test_set,
            limit=test_limit,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
        )

    model = build_model(
        use_scheme_efficiency_features=args.use_scheme_efficiency_features,
        tactical_assignment_bias=args.tactical_assignment_bias,
        tactical_bias_alpha_init=args.tactical_bias_alpha_init,
        assignment_temperature=args.assignment_temperature,
        redistribution_mode=args.redistribution_mode,
        redistribution_beta=args.redistribution_beta,
        redistribution_iterations=args.redistribution_iterations,
        redistribution_eta=args.redistribution_eta,
    ).to(device)
    initialization_checkpoint = None
    if args.initialize_from is not None:
        initialization_checkpoint = torch.load(
            args.initialize_from,
            map_location=device,
            weights_only=False,
        )
        initialize_model_weights(model, initialization_checkpoint)
        print(f"initialized from : {args.initialize_from}")
    print(
        "scheme efficiency:",
        "ENABLED" if args.use_scheme_efficiency_features else "DISABLED",
    )
    print(
        "tactical assignment bias:",
        args.tactical_assignment_bias,
        f"(alpha_init={args.tactical_bias_alpha_init:.6f})",
    )
    print("assignment temperature:", f"{args.assignment_temperature:.6f}")
    print(
        "redistribution:",
        args.redistribution_mode,
        f"beta={args.redistribution_beta:.3f}",
        f"T={args.redistribution_iterations}",
        f"eta={args.redistribution_eta:.3f}",
    )

    if args.no_conflict:
        conflict_regularizer = None
        print("conflict regularizer: DISABLED")
    else:
        surrogate, temperature = load_frozen_conflict_gnn(
            TRAIN_CONFIG.conflict_checkpoint,
            device=device,
            temperature_mode=TRAIN_CONFIG.conflict_temperature_mode,
        )

        map_cfg = DATA_CONFIG.maps
        map_context = load_map_context(
            map_cfg.friendly_map,
            map_cfg.enemy_map,
            friendly_step=map_cfg.friendly_grid_step,
            enemy_step=map_cfg.enemy_grid_step,
            margin=map_cfg.grid_margin,
        )

        conflict_regularizer = FrozenConflictRegularizer(
            surrogate=surrogate,
            num_weapons=MODEL_CONFIG.num_weapons,
            map_height=map_context.height,
            map_width=map_context.width,
            temperature=temperature,
            route_activation_tau=TRAIN_CONFIG.route_activation_tau,
            max_routes_per_graph=TRAIN_CONFIG.max_conflict_routes_per_graph,
            route_sample_repeats=TRAIN_CONFIG.conflict_route_sample_repeats,
            sample_seed=TRAIN_CONFIG.conflict_sample_seed,
        ).to(device)

        print(
            "conflict regularizer:",
            f"checkpoint={TRAIN_CONFIG.conflict_checkpoint}",
        )
        print(
            "surrogate temperature:",
            f"{temperature:.6f} ({TRAIN_CONFIG.conflict_temperature_mode})",
        )

    normalizer = EMALossNormalizer(
        keys=T1_CONFIG.ema_keys,
        beta=args.ema_beta,
        eps=T1_CONFIG.ema_eps,
        clip_min=T1_CONFIG.ema_clip_min,
        clip_max=T1_CONFIG.ema_clip_max,
    ).to(device)

    criterion = DeepWTALossT1(
        num_weapons=MODEL_CONFIG.num_weapons,
        num_schemes=MODEL_CONFIG.num_schemes,
        inventory_max=MODEL_CONFIG.inventory_max,
        conflict_regularizer=conflict_regularizer,
        demand_tolerance=TRAIN_CONFIG.demand_tolerance,
        normalizer=normalizer,
    ).to(device)

    constraint_controller = RobustConstraintControllerV2(
        lambda_demand=TRAIN_CONFIG.full_weights.demand,
        lambda_inventory=TRAIN_CONFIG.full_weights.inventory_pre,
        target_mode=args.constraint_target_mode,
        balanced_window=args.balanced_window,
        robust_window=args.robust_window,
        hysteresis_low=args.hysteresis_low,
        hysteresis_high=args.hysteresis_high,
        eta_up=args.dual_eta_up,
        eta_down=args.dual_eta_down,
        update_interval=args.controller_update_interval,
        full_warmup_epochs=args.controller_warmup_epochs,
        soft_gate_floor=args.soft_gate_floor,
    )

    if initialization_checkpoint is not None:
        if initialization_checkpoint.get("normalizer_state"):
            normalizer.load_state_dict(
                initialization_checkpoint["normalizer_state"]
            )
        if initialization_checkpoint.get("constraint_controller_state"):
            constraint_controller.load_state_dict(
                initialization_checkpoint["constraint_controller_state"]
            )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=TRAIN_CONFIG.weight_decay,
    )

    min_ratio = TRAIN_CONFIG.lr_min / max(args.lr, 1e-12)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: cosine_warmup_factor(
            e,
            total_epochs=args.epochs,
            warmup_epochs=TRAIN_CONFIG.warmup_epochs,
            min_ratio=min_ratio,
        ),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    checkpoint_dir = TRAIN_CONFIG.checkpoint_dir / args.run_name
    result_dir = TRAIN_CONFIG.result_dir / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoint_dir / "best.pt"
    best_full_path = checkpoint_dir / "best_full_feasible.pt"
    last_path = checkpoint_dir / "last.pt"
    log_path = result_dir / "training_log.csv"
    final_metrics_path = result_dir / "final_metrics.json"

    start_epoch = 1
    phase = (
        "FULL_OPT"
        if args.smoke_full or initialization_checkpoint is not None
        else "FEASIBILITY"
    )
    phase_start_epoch = 1
    ema_shortage = math.nan
    best_score = math.inf
    best_epoch = -1
    stale_epochs = 0

    best_full_score = math.inf
    best_full_epoch = -1
    full_stale_epochs = 0

    if args.smoke_full and initialization_checkpoint is None:
        constraint_controller.start_full_phase()

    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            constraint_controller=constraint_controller,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        phase = str(checkpoint["phase"])
        phase_start_epoch = int(checkpoint["phase_start_epoch"])
        ema_shortage = float(checkpoint["ema_shortage"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        best_full_score = float(
            checkpoint.get("best_full_score", math.inf)
        )
        best_full_epoch = int(
            checkpoint.get("best_full_epoch", -1)
        )
        full_stale_epochs = int(
            checkpoint.get("full_stale_epochs", 0)
        )
        print(
            f"resumed from epoch {start_epoch - 1}, "
            f"phase={phase}, best_epoch={best_epoch}"
        )
    elif log_path.exists():
        log_path.unlink()

    for epoch in range(start_epoch, args.epochs + 1):
        weights_now = runtime_phase_weights(
            phase,
            constraint_controller,
            tactical_profile,
        )

        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            sampler=train_sampler,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            phase=phase,
            phase_start_epoch=phase_start_epoch,
            amp_enabled=amp_enabled,
            requested_loss_mode=args.loss_mode,
            ema_ramp_epochs=args.ema_ramp_epochs,
            runtime_weights=weights_now,
        )

        scheduler.step()

        shortage_now = train_metrics["demand"]
        if math.isnan(ema_shortage):
            ema_shortage = shortage_now
        else:
            ema_shortage = (
                TRAIN_CONFIG.ema_beta * ema_shortage
                + (1.0 - TRAIN_CONFIG.ema_beta) * shortage_now
            )

        val_metrics = evaluate(
            model,
            criterion,
            val_loader,
            device,
            split="val",
        )
        score = validation_score(val_metrics)

        # BALANCED statistics define the FULL constraint envelope.
        if phase == "BALANCED":
            constraint_controller.observe_balanced(val_metrics)

        # Robust FULL feedback uses a rolling median, hysteresis and
        # anti-windup. The update affects the NEXT epoch.
        if phase == "FULL_OPT":
            constraint_controller.observe_full(val_metrics)
            constraint_controller.update_if_due()

        full_feasible = (
            phase == "FULL_OPT"
            and constraint_controller.selection_feasible(val_metrics)
        )
        current_full_score = (
            full_selection_score(
                val_metrics,
                tactical_profile,
            )
            if full_feasible
            else math.inf
        )

        controller_snapshot = constraint_controller.snapshot()

        row = {
            "run_name": args.run_name,
            "seed": args.seed,
            "loss_mode": args.loss_mode,
            "training_variant": "T2B_ROBUST_BALANCED",
            "tactical_style": args.tactical_style,
            "epoch": epoch,
            "phase": phase,
            "phase_start_epoch": phase_start_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "scheme_tau": train_metrics["scheme_tau"],
            "objective_scale": train_metrics["objective_scale"],
            "weight_demand": weights_now["demand"],
            "weight_inventory_pre": weights_now["inventory_pre"],
            "weight_ammo": weights_now["ammo"],
            "weight_crossing": weights_now["crossing"],
            "weight_distance": weights_now["distance"],
            "weight_interception": weights_now["interception"],
            "lambda_demand_next": controller_snapshot["lambda_demand"],
            "lambda_inventory_next": controller_snapshot["lambda_inventory"],
            "constraint_soft_gate_next": controller_snapshot["soft_gate"],
            "target_demand": controller_snapshot["target_demand"],
            "target_inventory_pre": controller_snapshot["target_inventory_pre"],
            "target_satisfaction": controller_snapshot["target_satisfaction"],
            "robust_demand": controller_snapshot["robust_demand"],
            "robust_inventory_pre": controller_snapshot["robust_inventory_pre"],
            "robust_satisfaction": controller_snapshot["robust_satisfaction"],
            "ema_demand_pressure": controller_snapshot["ema_demand_pressure"],
            "ema_inventory_pressure": controller_snapshot["ema_inventory_pressure"],
            "controller_safe_streak": controller_snapshot["safe_streak"],
            "controller_violation_streak": controller_snapshot["violation_streak"],
            "full_selection_feasible": float(full_feasible),
            "full_selection_score": current_full_score,
            "ema_demand_shortage": ema_shortage,
            "train_loss": train_metrics["loss"],
            "train_raw_reference_loss": train_metrics["raw_reference_loss"],
            "train_ema_strength": train_metrics["ema_strength"],
            "train_ema_scale_ammo": train_metrics["ema_scale_ammo"],
            "train_ema_scale_crossing": train_metrics["ema_scale_crossing"],
            "train_ema_scale_distance": train_metrics["ema_scale_distance"],
            "train_ema_scale_interception": train_metrics["ema_scale_interception"],
            "train_norm_ammo": train_metrics["norm_ammo"],
            "train_norm_crossing": train_metrics["norm_crossing"],
            "train_norm_distance": train_metrics["norm_distance"],
            "train_norm_interception": train_metrics["norm_interception"],
            "train_demand": train_metrics["demand"],
            "train_inventory_pre": train_metrics["inventory_pre"],
            "train_projection_pressure": train_metrics["projection_pressure"],
            "train_scheme_entropy": train_metrics["scheme_entropy"],
            "train_ammo": train_metrics["ammo"],
            "train_interception": train_metrics["interception"],
            "train_distance": train_metrics["distance"],
            "train_crossing": train_metrics["crossing"],
            "train_demand_satisfaction": train_metrics["demand_satisfaction_rate"],
            "train_route_coverage": train_metrics["route_coverage"],
            "val_score": score,
            "val_loss": val_metrics["loss"],
            "val_demand": val_metrics["demand"],
            "val_inventory_pre": val_metrics["inventory_pre"],
            "val_inventory_post": val_metrics["inventory_post"],
            "val_projection_pressure": val_metrics["projection_pressure"],
            "val_scheme_entropy": val_metrics["scheme_entropy"],
            "val_ammo": val_metrics["ammo"],
            "val_interception": val_metrics["interception"],
            "val_distance": val_metrics["distance"],
            "val_crossing": val_metrics["crossing"],
            "val_demand_satisfaction": val_metrics["demand_satisfaction_rate"],
            "val_conflict_expected_count": val_metrics["conflict_expected_count"],
            "val_route_coverage": val_metrics["route_coverage"],
        }
        append_csv(log_path, row)

        print(
            f"epoch={epoch:03d} phase={phase:<11} "
            f"train_dem={train_metrics['demand']:.4f} "
            f"val_dem={val_metrics['demand']:.4f} "
            f"val_sat={val_metrics['demand_satisfaction_rate']:.4f} "
            f"val_cross={val_metrics['crossing']:.4f} "
            f"val_ammo={val_metrics['ammo']:.4f} "
            f"score={score:.4f} "
            f"ld={constraint_controller.lambda_demand:.1f} "
            f"gate={constraint_controller.soft_gate():.2f} "
            f"p={controller_snapshot['ema_demand_pressure']:.2f}"
        )

        improved = score < best_score - TRAIN_CONFIG.early_stop_min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model,
                criterion,
                optimizer,
                scheduler,
                scaler,
                epoch,
                phase,
                phase_start_epoch,
                ema_shortage,
                best_score,
                best_epoch,
                stale_epochs,
                args.run_name,
                args.seed,
                args.loss_mode,
                args.ema_beta,
                args.ema_ramp_epochs,
                args.tactical_style,
                tactical_profile,
                constraint_controller,
                best_full_score,
                best_full_epoch,
                full_stale_epochs,
            )
        else:
            stale_epochs += 1

        if full_feasible:
            improved_full = (
                current_full_score
                < best_full_score - TRAIN_CONFIG.early_stop_min_delta
            )
            if improved_full:
                best_full_score = current_full_score
                best_full_epoch = epoch
                full_stale_epochs = 0
                save_checkpoint(
                    best_full_path,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    phase,
                    phase_start_epoch,
                    ema_shortage,
                    best_score,
                    best_epoch,
                    stale_epochs,
                    args.run_name,
                    args.seed,
                    args.loss_mode,
                    args.ema_beta,
                    args.ema_ramp_epochs,
                    args.tactical_style,
                    tactical_profile,
                    constraint_controller,
                    best_full_score,
                    best_full_epoch,
                    full_stale_epochs,
                )
            else:
                full_stale_epochs += 1
        elif phase == "FULL_OPT":
            # Do not count infeasible epochs as evidence that the feasible
            # tactical optimum has converged.
            full_stale_epochs = 0

        next_phase, next_phase_start = maybe_promote_phase(
            phase,
            epoch,
            phase_start_epoch,
            ema_shortage,
        )
        if next_phase != phase:
            print(
                f"CURRICULUM: {phase} -> {next_phase} "
                f"(ema demand shortage={ema_shortage:.4f})"
            )
            phase = next_phase
            phase_start_epoch = next_phase_start
            stale_epochs = 0
            if phase == "FULL_OPT":
                targets = constraint_controller.start_full_phase()
                full_stale_epochs = 0
                print(
                    "AUTO CONSTRAINT TARGETS: "
                    f"demand<={targets.demand:.6g}, "
                    f"inventory<={targets.inventory_pre:.6g}, "
                    f"satisfaction>={targets.satisfaction:.4f}"
                )

        # last.pt stores the state that should be used at the NEXT epoch,
        # including any curriculum promotion that happened above.
        save_checkpoint(
            last_path,
            model,
            criterion,
            optimizer,
            scheduler,
            scaler,
            epoch,
            phase,
            phase_start_epoch,
            ema_shortage,
            best_score,
            best_epoch,
            stale_epochs,
            args.run_name,
            args.seed,
            args.loss_mode,
            args.ema_beta,
            args.ema_ramp_epochs,
            args.tactical_style,
            tactical_profile,
            constraint_controller,
            best_full_score,
            best_full_epoch,
            full_stale_epochs,
        )

        full_age = (
            epoch - phase_start_epoch + 1
            if phase == "FULL_OPT"
            else 0
        )
        if (
            phase == "FULL_OPT"
            and best_full_epoch >= 0
            and full_age >= 40
            and full_stale_epochs >= TRAIN_CONFIG.early_stop_patience
        ):
            print(
                f"early stop at epoch {epoch}; "
                f"best FULL-feasible epoch={best_full_epoch}, "
                f"score={best_full_score:.6f}"
            )
            break

    if not best_path.exists():
        raise RuntimeError("training ended without creating best.pt")

    selected_path = (
        best_full_path
        if best_full_path.exists()
        else best_path
    )
    selection_fallback = not best_full_path.exists()
    if selection_fallback:
        print(
            "WARNING: no FULL-feasible checkpoint satisfied the adaptive "
            "selection targets; falling back to global raw-score best.pt"
        )

    best_checkpoint = torch.load(
        selected_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state"])
    if best_checkpoint.get("normalizer_state"):
        criterion.normalizer.load_state_dict(best_checkpoint["normalizer_state"])
    if best_checkpoint.get("constraint_controller_state"):
        constraint_controller.load_state_dict(
            best_checkpoint["constraint_controller_state"]
        )

    val_final = evaluate(
        model,
        criterion,
        val_loader,
        device,
        split="val-final",
    )
    test_final = None
    if test_loader is not None:
        test_final = evaluate(
            model,
            criterion,
            test_loader,
            device,
            split="test",
        )

    report = {
        "run_name": args.run_name,
        "seed": int(args.seed),
        "training_variant": "T2B_ROBUST_BALANCED",
        "tactical_style": args.tactical_style,
        "tactical_profile": tactical_profile.as_dict(),
        "tactical_budget": float(TACTICAL_BUDGET),
        "tactical_effective_weights": tactical_profile.effective_dict(),
        "tactical_profile_description": tactical_profile.description,
        "loss_mode": args.loss_mode,
        "ema_beta": float(args.ema_beta),
        "ema_ramp_epochs": int(args.ema_ramp_epochs),
        "ema_scales": criterion.normalizer.snapshot(),
        "constraint_controller": constraint_controller.snapshot(),
        "selected_checkpoint": str(selected_path),
        "selection_fallback_to_global_best": bool(selection_fallback),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_score": float(best_checkpoint["best_score"]),
        "dataset_version": TRAIN_CONFIG.dataset_version,
        "conflict_checkpoint": (
            None if args.no_conflict else str(TRAIN_CONFIG.conflict_checkpoint)
        ),
        "conflict_temperature_mode": (
            None if args.no_conflict else TRAIN_CONFIG.conflict_temperature_mode
        ),
        "validation": val_final,
        "test": test_final,
        "test_data_accessed": not args.validation_only,
    }
    final_metrics_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"global best     : {best_path}")
    print(f"FULL feasible   : {best_full_path}")
    print(f"selected        : {selected_path}")
    print(f"last checkpoint : {last_path}")
    print(f"training log    : {log_path}")
    print(f"final metrics   : {final_metrics_path}")


if __name__ == "__main__":
    main()
