from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from typing import Callable

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.dataset_config import CONFIG, ConflictDataConfig, MapConfig, SplitConfig, WTADataConfig
from configs.dataset.benchmark_scales import get_benchmark_scale
from deepwta.data.manifest import relative_to_project, sha256_file
from deepwta.data.map_data import load_map_context
from deepwta.data.trajectory_data import generate_conflict_graph
from deepwta.data.wta_data import generate_wta_graph


_WORKER_CONTEXT = None
_WORKER_CFG = None
_WORKER_KIND = None


def _init_worker(kind: str, map_cfg: MapConfig, data_cfg) -> None:
    global _WORKER_CONTEXT, _WORKER_CFG, _WORKER_KIND
    _WORKER_KIND = kind
    _WORKER_CFG = data_cfg
    _WORKER_CONTEXT = load_map_context(
        map_cfg.friendly_map,
        map_cfg.enemy_map,
        map_cfg.friendly_grid_step,
        map_cfg.enemy_grid_step,
        map_cfg.grid_margin,
    )


def _worker(task: tuple[int, int, str]):
    index, seed, split = task
    scenario_id = f"{_WORKER_KIND}_{split}_{index:06d}"
    if _WORKER_KIND == "wta":
        return generate_wta_graph(_WORKER_CFG, _WORKER_CONTEXT, seed, scenario_id)
    return generate_conflict_graph(_WORKER_CFG, _WORKER_CONTEXT, seed, scenario_id)


def _split_seed(base_seed: int, split: str, index: int) -> int:
    offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}[split]
    return int(base_seed + offset + index)


def _save_split(
    kind: str,
    split: str,
    count: int,
    output_dir: Path,
    shard_size: int,
    workers: int,
    map_cfg: MapConfig,
    data_cfg,
    base_seed: int,
) -> list[str]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for old_file in split_dir.glob("part_*.pt"):
        old_file.unlink()

    tasks = [(i, _split_seed(base_seed, split, i), split) for i in range(count)]
    shard_files: list[str] = []
    buffer = []
    part = 0

    def flush() -> None:
        nonlocal part
        if not buffer:
            return
        file_path = split_dir / f"part_{part:04d}.pt"
        torch.save(list(buffer), file_path)
        shard_files.append(str(file_path.relative_to(PROJECT_ROOT)))
        buffer.clear()
        part += 1

    if workers <= 1:
        _init_worker(kind, map_cfg, data_cfg)
        iterator = map(_worker, tasks)
        for graph in tqdm(iterator, total=count, desc=f"{kind}:{split}"):
            buffer.append(graph)
            if len(buffer) >= shard_size:
                flush()
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(kind, map_cfg, data_cfg),
        ) as executor:
            iterator = executor.map(_worker, tasks, chunksize=4)
            for graph in tqdm(iterator, total=count, desc=f"{kind}:{split}"):
                buffer.append(graph)
                if len(buffer) >= shard_size:
                    flush()
    flush()
    return shard_files


def _manifest_path(dataset_name: str) -> Path:
    return CONFIG.splits_dir / f"{dataset_name}.json"


def _write_manifest(
    kind: str,
    dataset_name: str,
    scale: str | None,
    output_dir: Path,
    split_files: dict,
    map_cfg,
    split_cfg,
    data_cfg,
) -> Path:
    CONFIG.splits_dir.mkdir(parents=True, exist_ok=True)
    map_hash = {
        "friendly_map_sha256": sha256_file(map_cfg.friendly_map),
        "enemy_map_sha256": sha256_file(map_cfg.enemy_map),
    }
    generation_config = {
        "map": {
            "friendly_map": relative_to_project(map_cfg.friendly_map),
            "enemy_map": relative_to_project(map_cfg.enemy_map),
            "friendly_grid_step": map_cfg.friendly_grid_step,
            "enemy_grid_step": map_cfg.enemy_grid_step,
            "grid_margin": map_cfg.grid_margin,
        },
        "split": asdict(split_cfg),
        "wta": asdict(data_cfg) if kind == "wta" else None,
        "conflict": asdict(data_cfg) if kind == "conflict" else None,
    }
    manifest = {
        "dataset": kind,
        "dataset_name": dataset_name,
        "scale": scale,
        "seed": int(split_cfg.base_seed),
        "map_hash": map_hash,
        "generation_config": generation_config,
        "version": CONFIG.version,
        "root": relative_to_project(output_dir),
        "map": {
            "friendly_map": relative_to_project(map_cfg.friendly_map),
            "enemy_map": relative_to_project(map_cfg.enemy_map),
            "friendly_grid_step": map_cfg.friendly_grid_step,
            "enemy_grid_step": map_cfg.enemy_grid_step,
            "grid_margin": map_cfg.grid_margin,
        },
        "split": {
            "train": split_cfg.train,
            "val": split_cfg.val,
            "test": split_cfg.test,
            "base_seed": split_cfg.base_seed,
            "shard_size": split_cfg.shard_size,
        },
        "files": split_files,
        "config": data_cfg.__dict__,
    }
    path = _manifest_path(dataset_name)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=list)
    print(f"manifest: {path}")
    return path


def _densify_map_candidates(map_cfg: MapConfig, fixed_bases: int, fixed_targets: int) -> MapConfig:
    """Increase candidate density without changing either source map or mask semantics."""
    friendly_step = int(map_cfg.friendly_grid_step)
    enemy_step = int(map_cfg.enemy_grid_step)
    while True:
        context = load_map_context(
            map_cfg.friendly_map,
            map_cfg.enemy_map,
            friendly_step,
            enemy_step,
            map_cfg.grid_margin,
        )
        bases_ok = len(context.base_candidates) >= fixed_bases
        targets_ok = len(context.target_candidates) >= fixed_targets
        if bases_ok and targets_ok:
            return replace(
                map_cfg,
                friendly_grid_step=friendly_step,
                enemy_grid_step=enemy_step,
            )
        if not bases_ok and friendly_step > 1:
            friendly_step -= 1
        if not targets_ok and enemy_step > 1:
            enemy_step -= 1
        if (not bases_ok and friendly_step == 1) or (not targets_ok and enemy_step == 1):
            # Re-evaluate step=1 once before declaring the unchanged maps too small.
            final_context = load_map_context(
                map_cfg.friendly_map, map_cfg.enemy_map, 1, 1, map_cfg.grid_margin
            )
            if len(final_context.base_candidates) < fixed_bases or len(final_context.target_candidates) < fixed_targets:
                raise RuntimeError(
                    "fixed scale exceeds map land candidates even at grid step 1: "
                    f"bases={fixed_bases}/{len(final_context.base_candidates)}, "
                    f"targets={fixed_targets}/{len(final_context.target_candidates)}"
                )
            friendly_step = 1 if not bases_ok else friendly_step
            enemy_step = 1 if not targets_ok else enemy_step


def _validate_config_schema() -> None:
    required = {
        "wta": (
            "min_size_ratio",
            "max_size_ratio",
            "min_bases",
            "max_bases",
            "min_targets",
            "max_targets",
        ),
        "conflict": (
            "min_size_ratio",
            "max_size_ratio",
            "min_bases",
            "max_bases",
            "min_targets",
            "max_targets",
        ),
    }
    for section_name, fields in required.items():
        section = getattr(CONFIG, section_name)
        missing = [name for name in fields if not hasattr(section, name)]
        if missing:
            raise RuntimeError(
                f"configs/dataset_config.py is incompatible with {CONFIG.version}: "
                f"{section_name} is missing {missing}. "
                "Please overwrite configs/dataset_config.py with the v5 version."
            )

        min_ratio = float(section.min_size_ratio)
        max_ratio = float(section.max_size_ratio)
        if not (0.0 < min_ratio <= max_ratio):
            raise RuntimeError(
                f"invalid {section_name} N/M ratio range: "
                f"{min_ratio} <= N/M <= {max_ratio}"
            )


def _build_configs(args):
    map_cfg = replace(
        CONFIG.maps,
        friendly_map=Path(args.friendly_map).resolve() if args.friendly_map else CONFIG.maps.friendly_map,
        enemy_map=Path(args.enemy_map).resolve() if args.enemy_map else CONFIG.maps.enemy_map,
    )
    split_cfg = replace(
        CONFIG.split,
        train=args.train if args.train is not None else CONFIG.split.train,
        val=args.val if args.val is not None else CONFIG.split.val,
        test=args.test if args.test is not None else CONFIG.split.test,
        base_seed=args.seed if args.seed is not None else CONFIG.split.base_seed,
        shard_size=args.shard_size if args.shard_size is not None else CONFIG.split.shard_size,
    )
    return map_cfg, split_cfg


def generate_main(args) -> None:
    map_cfg, split_cfg = _build_configs(args)
    if args.num_weapons is not None and int(args.num_weapons) != int(CONFIG.wta.num_weapons):
        raise ValueError(
            f"num_weapons is the frozen weapon-type dimension and must remain "
            f"{CONFIG.wta.num_weapons}; use --total-weapons for inventory scale"
        )
    dataset_name = args.dataset_name if args.dataset_name is not None else f"wta_{CONFIG.version}"
    if args.total_weapons is not None and int(args.total_weapons) <= 0:
        raise ValueError("total_weapons must be positive")
    if (
        args.zero_crossing_matching_attempts is not None
        and int(args.zero_crossing_matching_attempts) <= 0
    ):
        raise ValueError("zero_crossing_matching_attempts must be positive")
    if args.scale is not None:
        scale = get_benchmark_scale(args.scale)
        received_physical = (
            args.total_weapons,
            args.fixed_targets,
            args.fixed_bases,
        )
        expected_physical = (
            scale.total_weapons,
            scale.targets,
            scale.bases,
        )
        if received_physical != expected_physical:
            raise ValueError(
                f"scale {args.scale!r} requires total/targets/bases="
                f"{expected_physical}, got {received_physical}"
            )
        # Ordinary formal benchmark generation keeps the frozen canonical name.
        # The zero-crossing diagnostic branch may use a distinct name while
        # preserving exactly the same physical scale contract.
        if (
            dataset_name != scale.dataset_name
            and not bool(args.require_zero_crossing_witness)
        ):
            raise ValueError(
                f"scale {args.scale!r} requires dataset_name={scale.dataset_name!r} "
                "unless --require-zero-crossing-witness is enabled"
            )
    manifest_path = _manifest_path(dataset_name)
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"manifest already exists: {manifest_path}; pass --overwrite to replace this dataset"
        )
    if args.fixed_targets is not None:
        map_cfg = _densify_map_candidates(
            map_cfg,
            int(args.fixed_bases if args.fixed_bases is not None else args.fixed_targets),
            int(args.fixed_targets),
        )
    data_cfg = replace(
        CONFIG.wta,
        num_weapons=CONFIG.wta.num_weapons,
        num_schemes=args.num_schemes if args.num_schemes is not None else CONFIG.wta.num_schemes,
        reachability_rate=(
            args.reachability_rate if args.reachability_rate is not None else CONFIG.wta.reachability_rate
        ),
        min_reachable_bases=(
            args.min_reachable_bases if args.min_reachable_bases is not None else CONFIG.wta.min_reachable_bases
        ),
        level_resource_gap=(
            args.level_resource_gap if args.level_resource_gap is not None else CONFIG.wta.level_resource_gap
        ),
        fixed_targets=args.fixed_targets,
        fixed_bases=args.fixed_bases,
        total_weapons=args.total_weapons,
        benchmark_scale=args.scale,
        require_zero_crossing_witness=bool(args.require_zero_crossing_witness),
        zero_crossing_matching_attempts=(
            int(args.zero_crossing_matching_attempts)
            if args.zero_crossing_matching_attempts is not None
            else CONFIG.wta.zero_crossing_matching_attempts
        ),
    )
    output_dir = CONFIG.processed_dir / dataset_name
    split_files = {}
    for split, count in (("train", split_cfg.train), ("val", split_cfg.val), ("test", split_cfg.test)):
        split_files[split] = _save_split(
            "wta", split, count, output_dir, split_cfg.shard_size, args.workers,
            map_cfg, data_cfg, split_cfg.base_seed,
        )
    _write_manifest("wta", dataset_name, args.scale, output_dir, split_files, map_cfg, split_cfg, data_cfg)


def generate_conflict(args) -> None:
    map_cfg, split_cfg = _build_configs(args)
    dataset_name = f"conflict_{CONFIG.version}"
    manifest_path = _manifest_path(dataset_name)
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"manifest already exists: {manifest_path}; pass --overwrite to replace this dataset"
        )
    default_split = SplitConfig(train=20000, val=2500, test=2500, base_seed=split_cfg.base_seed, shard_size=split_cfg.shard_size)
    if args.train is None and args.val is None and args.test is None:
        split_cfg = default_split

    data_cfg = replace(
        CONFIG.conflict,
        min_trajectories=(
            args.min_trajectories if args.min_trajectories is not None else CONFIG.conflict.min_trajectories
        ),
        max_trajectories=(
            args.max_trajectories if args.max_trajectories is not None else CONFIG.conflict.max_trajectories
        ),
        negative_to_positive_ratio=(
            args.negative_ratio if args.negative_ratio is not None else CONFIG.conflict.negative_to_positive_ratio
        ),
    )
    output_dir = CONFIG.processed_dir / f"conflict_{CONFIG.version}"
    split_files = {}
    for split, count in (("train", split_cfg.train), ("val", split_cfg.val), ("test", split_cfg.test)):
        split_files[split] = _save_split(
            "conflict", split, count, output_dir, split_cfg.shard_size, args.workers,
            map_cfg, data_cfg, split_cfg.base_seed,
        )
    _write_manifest("conflict", dataset_name, None, output_dir, split_files, map_cfg, split_cfg, data_cfg)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DeepWTA v5 datasets")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--friendly-map", type=str, default=None)
        p.add_argument("--enemy-map", type=str, default=None)
        p.add_argument("--train", type=int, default=None)
        p.add_argument("--val", type=int, default=None)
        p.add_argument("--test", type=int, default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--shard-size", type=int, default=None)
        p.add_argument("--workers", type=int, default=CONFIG.workers)
        p.add_argument("--overwrite", action="store_true")

    main = sub.add_parser("main", help="generate DeepWTA WTA graphs")
    common(main)
    main.add_argument("--num-weapons", type=int, default=None)
    main.add_argument("--total-weapons", type=int, default=None)
    main.add_argument("--dataset-name", type=str, default=None)
    main.add_argument("--scale", choices=("small", "medium", "large", "xlarge"), default=None)
    main.add_argument("--fixed-targets", type=int, default=None)
    main.add_argument("--fixed-bases", type=int, default=None)
    main.add_argument("--num-schemes", type=int, default=None)
    main.add_argument("--reachability-rate", type=float, default=None)
    main.add_argument("--min-reachable-bases", type=int, default=None)
    main.add_argument("--level-resource-gap", type=float, default=None)
    main.add_argument(
        "--require-zero-crossing-witness",
        action="store_true",
        help=(
            "require the stored feasibility witness to have exact crossing count 0; "
            "all other v5 data-generation rules stay unchanged"
        ),
    )
    main.add_argument(
        "--zero-crossing-matching-attempts",
        type=int,
        default=None,
        help="alternative planar witness skeletons tried per ordinary scenario attempt",
    )
    main.set_defaults(func=generate_main)

    conflict = sub.add_parser("conflict", help="generate 2D route-crossing surrogate graphs")
    common(conflict)
    conflict.add_argument("--min-trajectories", type=int, default=None)
    conflict.add_argument("--max-trajectories", type=int, default=None)
    conflict.add_argument("--negative-ratio", type=float, default=None)
    conflict.set_defaults(func=generate_conflict)
    return parser


def main() -> None:
    _validate_config_schema()
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
