from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MapConfig:
    friendly_map: Path = PROJECT_ROOT / "data" / "maps" / "Friendly_Map.png"
    enemy_map: Path = PROJECT_ROOT / "data" / "maps" / "Enemy_Map.png"

    friendly_grid_step: int = 15
    enemy_grid_step: int = 10

    grid_margin: int = 30


@dataclass(frozen=True)
class SplitConfig:
    train: int = 6000
    val: int = 800
    test: int = 800

    base_seed: int = 2026

    shard_size: int = 128


@dataclass(frozen=True)
class WTADataConfig:

    # ==================================================
    # Fixed feature dimensions
    # ==================================================

    # weapon types
    num_weapons: int = 10

    # strike schemes
    num_schemes: int = 10


    # ==================================================
    # Resource / damage configuration
    # ==================================================

    demand_max: int = 8

    inventory_max: int = 80


    # ==================================================
    # Reachability
    # ==================================================

    reachability_rate: float = 0.97

    min_reachable_bases: int = 3


    # ==================================================
    # Strike plan resource gap
    # ==================================================

    level_resource_gap: float = 0.15


    damage_threshold_probs: tuple[
        float,
        float,
        float
    ] = (
        0.50,
        0.35,
        0.15
    )


    # ==================================================
    # Inventory generation
    # ==================================================

    reference_capacity_ratio: float = 0.72

    reserve_rate_min: float = 0.05

    reserve_rate_max: float = 0.55


    idle_inventory_lambda_min: float = 0.5

    idle_inventory_lambda_max: float = 3.5



    # ==================================================
    # Original random-scale training distribution
    #
    # Used for:
    # - model training
    # - generalization experiments
    #
    # Do NOT remove
    # ==================================================

    typical_sizes: tuple[
        tuple[int, int],
        ...
    ] = (
        (10, 10),
        (20, 10),
        (10, 20),
        (20, 20),

        (30, 20),
        (20, 30),

        (40, 20),
        (20, 40),

        (40, 40),

        (60, 30),
        (30, 60),

        (60, 60),

        (80, 40),
        (40, 80),

        (80, 80),
    )


    random_size_ratio: float = 0.25



    # ==================================================
    # Size constraints
    # ==================================================

    # Formal training distribution:
    #
    # 1/3 <= Bases / Targets <= 3
    #

    min_size_ratio: float = 1.0 / 3.0

    max_size_ratio: float = 3.0



    min_bases: int = 8

    max_bases: int = 80


    min_targets: int = 8

    max_targets: int = 80



    # ==================================================
    # NEW:
    # Fixed scalability benchmark mode
    #
    # None:
    #       original random generation
    #
    # integer:
    #       controlled experiment scale
    #
    # Example:
    #
    # Small:
    #   fixed_bases=30
    #   fixed_targets=20
    #
    # Large:
    #   fixed_bases=120
    #   fixed_targets=100
    #
    # ==================================================

    fixed_bases: int | None = None

    fixed_targets: int | None = None

    # Exact total inventory for fixed-scale benchmarks. This is intentionally
    # independent from num_weapons, which is the frozen K=10 weapon-type axis.
    total_weapons: int | None = None

    benchmark_scale: str | None = None



    # ==================================================
    # Optional diagnostic generation mode.
    #
    # False preserves the original v5 generator exactly.
    # True changes only the reference-witness construction: the stored
    # feasibility witness must additionally have zero exact route crossings.
    # All physical-data generation rules remain unchanged.
    # ==================================================

    require_zero_crossing_witness: bool = False

    # Number of alternative planar target-to-base skeletons tried within one
    # ordinary scenario-generation attempt before the whole attempt is rejected.
    zero_crossing_matching_attempts: int = 32

    # ==================================================

    max_scenario_attempts: int = 200



@dataclass(frozen=True)
class ConflictDataConfig:

    # ==================================================
    # Route conflict graph
    # ==================================================

    node_dim: int = 7

    edge_dim: int = 6



    min_bases: int = 10

    max_bases: int = 40


    min_targets: int = 10

    max_targets: int = 40



    min_size_ratio: float = 1.0 / 3.0

    max_size_ratio: float = 3.0



    min_trajectories: int = 24

    max_trajectories: int = 72



    route_reachability_rate: float = 0.97


    negative_to_positive_ratio: float = 1.25


    max_candidate_pairs: int = 5000


    max_graph_attempts: int = 30



@dataclass(frozen=True)
class DatasetConfig:

    maps: MapConfig = field(
        default_factory=MapConfig
    )


    split: SplitConfig = field(
        default_factory=SplitConfig
    )


    wta: WTADataConfig = field(
        default_factory=WTADataConfig
    )


    conflict: ConflictDataConfig = field(
        default_factory=ConflictDataConfig
    )


    processed_dir: Path = (
        PROJECT_ROOT /
        "data" /
        "processed"
    )


    splits_dir: Path = (
        PROJECT_ROOT /
        "data" /
        "splits"
    )


    workers: int = min(
        8,
        max(
            1,
            (os.cpu_count() or 2)-1
        )
    )


    version: str = "v5"



    def to_dict(self) -> dict:

        def normalize(value):

            if isinstance(value, Path):
                return str(value)


            if isinstance(value, tuple):

                return [
                    normalize(x)
                    for x in value
                ]


            if isinstance(value, list):

                return [
                    normalize(x)
                    for x in value
                ]


            if isinstance(value, dict):

                return {
                    k: normalize(v)
                    for k,v in value.items()
                }


            return value


        return normalize(
            asdict(self)
        )


CONFIG = DatasetConfig()
