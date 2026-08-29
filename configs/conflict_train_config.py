from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ConflictModelConfig:
    """Trajectory conflict surrogate model."""

    # 2D route node features and route-pair edge features.
    node_dim: int = 7
    edge_dim: int = 6

    # Lightweight model is sufficient for the geometric relation task.
    hidden_dim: int = 64
    heads: int = 4
    edge_mlp_dim: int = 128
    dropout: float = 0.10


@dataclass(frozen=True)
class ConflictTrainConfig:
    """Training configuration for TrajectoryConflictGNN."""

    dataset_version: str = "v5"

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    # Maximum training budget. Early stopping normally terminates earlier.
    epochs: int = 180
    batch_size: int = 16

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    lr_min: float = 1e-6

    grad_clip_norm: float = 5.0

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    # Edge classification is the primary task.
    edge_loss_weight: float = 1.0

    # Probability quality is important because DeepWTA consumes p_rs
    # directly as a continuous conflict propensity.
    brier_loss_weight: float = 0.30

    # Auxiliary graph-level crossing-rate consistency.
    graph_rate_loss_weight: float = 0.10

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------
    # Avoid terminating during the fast-learning stage.
    min_epochs: int = 50

    # Stop when validation quality has no meaningful improvement.
    early_stop_patience: int = 20

    # Improvements below this value are treated as numerical fluctuations.
    early_stop_min_delta: float = 2e-5

    # Near-optimal PR-AUC checkpoints may later be compared using
    # probability/calibration metrics instead of PR-AUC alone.
    checkpoint_pr_auc_tolerance: float = 1e-4

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    num_workers: int = 4
    shard_cache_size: int = 4
    seed: int = 2026
    use_amp: bool = True

    # ------------------------------------------------------------------
    # Probability calibration
    # ------------------------------------------------------------------
    # No artificial lower bound is imposed on the calibrated temperature.
    #
    # Existing finalize_conflict_gnn.py computes:
    #
    #   deployment_temperature =
    #       max(calibrated_temperature, min_deployment_temperature)
    #
    # Setting this to 0 disables the previous T >= 1.20 rule and makes
    # the calibrated validation-set temperature the default downstream
    # temperature.
    min_deployment_temperature: float = 0.0

    # These temperatures are only candidates for downstream ablation.
    # They are NOT used to overwrite the calibrated temperature.
    temperature_ablation_values: Tuple[float, ...] = (
        1.0,
        1.2,
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints" / "conflict_gnn"
    result_dir: Path = PROJECT_ROOT / "results" / "conflict_gnn"


MODEL_CONFIG = ConflictModelConfig()
TRAIN_CONFIG = ConflictTrainConfig()