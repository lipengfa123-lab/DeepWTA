from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DeepWTASolverConfig:
    """Configuration for DeepWTA-Standard.

    Fixed passes, target caps and trial caps are the normal stopping criteria;
    solution quality never depends on a wall-clock deadline in formal runs.
    """

    route_mass_rho: float = 0.90
    route_k_min: int = 2
    route_k_max: int = 8
    route_softmax_tau: float = 0.25

    reuse_active_routes_first: bool = True
    bulk_fill: bool = True

    # Neural-to-integer interface. ``floor`` is the formal baseline; the
    # residual-aware modes are Phase-4A inference-only experiments.
    integer_seed_mode: str = "floor"
    seed_preserving_fallback: bool = False

    weight_ammo: float = 10.0
    weight_crossing: float = 80.0
    weight_distance: float = 30.0
    weight_interception: float = 10.0

    max_passes: int = 4
    min_relative_improvement: float = 0.002
    stall_passes: int = 2
    max_hotspot_targets: int = 10
    max_route_consolidation_trials_per_target: int = 16
    max_transfer_trials_per_target: int = 24

    enable_rare_scheme_change: bool = False
    scheme_probability_gap: float = 0.08
    max_scheme_targets: int = 3

    enable_small_medium_topology_polish: bool = True
    topology_polish_passes: int = 2
    topology_polish_max_targets: int = 20
    topology_polish_trials_per_target: int = 48
    topology_min_crossing_reduction: int = 1
    topology_max_avg_distance_increase: float = 0.0

    enable_global_distance_polish: bool = True
    distance_polish_passes: int = 1
    distance_polish_max_targets: int = 16
    distance_polish_route_trials_per_target: int = 24
    distance_polish_shot_trials_per_target: int = 16
    distance_min_avg_improvement: float = 1e-5
    distance_max_crossing_increase: int = 0

    # Determinism hotfix. Keep True for all formal experiments.
    deterministic_search: bool = True

    # Optional wall-clock controls are ignored while deterministic_search=True.
    runtime_budget_ms: float | None = None
    auto_runtime_budget: bool = False
    small_budget_ms: float = 100.0
    medium_budget_ms: float = 300.0
    large_budget_ms: float = 600.0
    xlarge_budget_ms: float = 1200.0
    topology_small_budget_ms: float = 20.0
    topology_medium_budget_ms: float = 35.0
    distance_small_budget_ms: float = 10.0
    distance_medium_budget_ms: float = 15.0
    distance_large_budget_ms: float = 20.0
    distance_xlarge_budget_ms: float = 25.0

    fallback_to_construction: bool = True
    tol: float = 1e-7

    def validate(self) -> None:
        if self.integer_seed_mode not in {
            "floor", "ra_r", "ra_s", "ra_n", "gb_r", "gb_s", "gb_sr",
        }:
            raise ValueError(
                "integer_seed_mode must be one of: "
                "floor, ra_r, ra_s, ra_n, gb_r, gb_s, gb_sr"
            )
        weights = (
            self.weight_ammo, self.weight_crossing,
            self.weight_distance, self.weight_interception,
        )
        if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
            raise ValueError("solver tactical weights must be non-negative with a positive sum")
        if not (0.0 < self.route_mass_rho <= 1.0):
            raise ValueError("route_mass_rho must be in (0,1]")
        if self.route_k_min < 1:
            raise ValueError("route_k_min must be >= 1")
        if self.route_k_max < self.route_k_min:
            raise ValueError("route_k_max must be >= route_k_min")
        if self.route_softmax_tau <= 0.0:
            raise ValueError("route_softmax_tau must be > 0")
        if self.max_passes < 0:
            raise ValueError("max_passes must be non-negative")
        if self.stall_passes < 1:
            raise ValueError("stall_passes must be >= 1")
        if self.max_route_consolidation_trials_per_target < 1:
            raise ValueError("max_route_consolidation_trials_per_target must be >= 1")
        if self.max_transfer_trials_per_target < 1:
            raise ValueError("max_transfer_trials_per_target must be >= 1")
        if self.topology_polish_passes < 0:
            raise ValueError("topology_polish_passes must be non-negative")
        if self.topology_polish_max_targets < 1:
            raise ValueError("topology_polish_max_targets must be >= 1")
        if self.topology_polish_trials_per_target < 1:
            raise ValueError("topology_polish_trials_per_target must be >= 1")
        if self.topology_min_crossing_reduction < 1:
            raise ValueError("topology_min_crossing_reduction must be >= 1")
        if self.topology_max_avg_distance_increase < 0.0:
            raise ValueError("topology_max_avg_distance_increase must be non-negative")
        if self.distance_polish_passes < 0:
            raise ValueError("distance_polish_passes must be non-negative")
        if self.distance_polish_max_targets < 1:
            raise ValueError("distance_polish_max_targets must be >= 1")
        if self.distance_polish_route_trials_per_target < 1:
            raise ValueError("distance_polish_route_trials_per_target must be >= 1")
        if self.distance_polish_shot_trials_per_target < 1:
            raise ValueError("distance_polish_shot_trials_per_target must be >= 1")
        if self.distance_min_avg_improvement < 0.0:
            raise ValueError("distance_min_avg_improvement must be non-negative")
        if self.distance_max_crossing_increase < 0:
            raise ValueError("distance_max_crossing_increase must be >= 0")

    @staticmethod
    def size_bucket(num_bases: int, num_targets: int) -> str:
        size = max(int(num_bases), int(num_targets))
        if size <= 20:
            return "small"
        if size <= 40:
            return "medium"
        if size <= 60:
            return "large"
        return "xlarge"

    def budget_seconds(self, num_bases: int, num_targets: int) -> float | None:
        if self.deterministic_search:
            return None
        if self.runtime_budget_ms is not None:
            return max(float(self.runtime_budget_ms), 0.0) / 1000.0
        if not self.auto_runtime_budget:
            return None
        value = {
            "small": self.small_budget_ms,
            "medium": self.medium_budget_ms,
            "large": self.large_budget_ms,
            "xlarge": self.xlarge_budget_ms,
        }[self.size_bucket(num_bases, num_targets)]
        return max(float(value), 0.0) / 1000.0

    def as_dict(self) -> dict:
        return asdict(self)
