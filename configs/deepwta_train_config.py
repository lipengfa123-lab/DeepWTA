from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from configs.dataset_config import CONFIG as DATA_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PhaseWeights:
    demand: float
    inventory_pre: float
    scheme_entropy: float
    ammo: float
    interception: float
    distance: float
    crossing: float

    def as_dict(self) -> dict[str, float]:
        return {
            "demand": self.demand,
            "inventory_pre": self.inventory_pre,
            "scheme_entropy": self.scheme_entropy,
            "ammo": self.ammo,
            "interception": self.interception,
            "distance": self.distance,
            "crossing": self.crossing,
        }


@dataclass(frozen=True)
class DeepWTATrainConfig:
    dataset_version: str = "v5"

    # Main optimization.
    epochs: int = 180
    batch_size: int = 4
    grad_accum_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0

    warmup_epochs: int = 5
    lr_min: float = 5e-6

    # By default use a large stochastic subset each epoch. Across 180 epochs this
    # repeatedly covers the full 6000-graph training corpus while keeping runtime
    # practical with the frozen trajectory surrogate in the loop.
    train_graphs_per_epoch: int = 3000
    val_graph_limit: int = 800
    test_graph_limit: int = 800

    num_workers: int = 4
    shard_cache_size: int = 4
    seed: int = 2026
    use_amp: bool = True
    deterministic: bool = True

    # Scheme relaxation.
    scheme_tau_start: float = 1.20
    scheme_tau_end: float = 0.40
    full_phase_gumbel_hard: bool = True

    # Curriculum promotion.
    ema_beta: float = 0.90
    feasibility_min_epochs: int = 15
    feasibility_max_epochs: int = 35
    feasibility_shortage_threshold: float = 0.040

    balanced_min_epochs: int = 25
    balanced_max_end_epoch: int = 75
    balanced_shortage_threshold: float = 0.025

    objective_ramp_epochs: int = 15

    feasibility_weights: PhaseWeights = field(
        default_factory=lambda: PhaseWeights(
            demand=12.0,
            inventory_pre=3.0,
            scheme_entropy=0.0,
            ammo=0.0,
            interception=0.0,
            distance=0.0,
            crossing=0.0,
        )
    )
    balanced_weights: PhaseWeights = field(
        default_factory=lambda: PhaseWeights(
            demand=12.0,
            inventory_pre=3.0,
            scheme_entropy=0.04,
            ammo=0.15,
            interception=0.08,
            distance=0.08,
            crossing=0.20,
        )
    )
    full_weights: PhaseWeights = field(
        default_factory=lambda: PhaseWeights(
            demand=12.0,
            inventory_pre=3.0,
            scheme_entropy=0.06,
            ammo=0.22,
            interception=0.10,
            distance=0.10,
            crossing=0.35,
        )
    )

    demand_tolerance: float = 0.05

    # Frozen Conflict GNN.
    conflict_checkpoint: Path = (
        PROJECT_ROOT
        / "checkpoints"
        / "conflict_gnn"
        / "best_calibrated.pt"
    )
    conflict_temperature_mode: str = "calibrated"
    route_activation_tau: float = 1.0

    # Keep each surrogate call close to its training graph-size distribution.
    max_conflict_routes_per_graph: int = 72
    conflict_route_sample_repeats: int = 1
    conflict_sample_seed: int = 3407

    # Model selection / early stop. Validation always uses hard scheme decisions
    # and FULL_OPT weights so scores remain comparable across curriculum phases.
    early_stop_patience: int = 25
    early_stop_min_delta: float = 1e-4
    min_epochs_before_stop: int = 85
    validation_infeasibility_weight: float = 2.0

    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints" / "deepwta"
    result_dir: Path = PROJECT_ROOT / "results" / "deepwta"

    @property
    def dataset_dir(self) -> Path:
        return (
            DATA_CONFIG.processed_dir
            / f"wta_{self.dataset_version}"
        )


TRAIN_CONFIG = DeepWTATrainConfig()
