from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def add_project_paths() -> None:
    import sys
    for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def group_id(index: int) -> str:
    return ("A", "B", "C")[int(index) % 3]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path, name: str):
    path = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_solution_npz(
    output_dir: Path,
    *,
    method: str,
    scenario_index: int,
    scenario_id: str,
    selected_scheme,
    allocation,
    selected_scheme_raw=None,
    allocation_raw=None,
    extra: dict | None = None,
) -> Path:
    out = output_dir / method
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"idx_{int(scenario_index):06d}.npz"
    payload = {
        "scenario_index": np.asarray(int(scenario_index), dtype=np.int64),
        "scenario_id": np.asarray(str(scenario_id)),
        "group_id": np.asarray(group_id(scenario_index)),
        "selected_scheme": np.asarray(selected_scheme, dtype=np.int64),
        "allocation": np.asarray(allocation, dtype=np.int64),
    }
    if selected_scheme_raw is not None:
        payload["selected_scheme_raw"] = np.asarray(selected_scheme_raw, dtype=np.int64)
    if allocation_raw is not None:
        payload["allocation_raw"] = np.asarray(allocation_raw, dtype=np.int64)
    if extra:
        payload["metadata_json"] = np.asarray(json.dumps(extra, ensure_ascii=False))
    np.savez_compressed(path, **payload)
    return path


def save_input_npz(output_dir: Path, instance, *, index: int, dataset_name: str, split: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"idx_{int(index):06d}.npz"
    payload = {
        "scenario_index": np.asarray(int(index), dtype=np.int64),
        "scenario_id": np.asarray(str(instance.instance_id)),
        "group_id": np.asarray(group_id(index)),
        "dataset_name": np.asarray(str(dataset_name)),
        "split": np.asarray(str(split)),
        "scenario_seed": np.asarray(int(instance.seed), dtype=np.int64),
        "base_coords": np.asarray(instance.base_coords),
        "target_coords": np.asarray(instance.target_coords),
        "B_inventory": np.asarray(instance.inventory),
        "U_scheme_demand": np.asarray(instance.scheme_demand),
        "G_damage_level": np.asarray(instance.damage_level),
        "G_th_damage_threshold": np.asarray(instance.damage_threshold),
        "Q_intercept_prob": np.asarray(instance.intercept_prob),
        "D_distance": np.asarray(instance.distance),
        "X_reachable": np.asarray(instance.reachable),
    }
    if instance.reference_scheme is not None:
        payload["reference_scheme_AUDIT_ONLY"] = np.asarray(instance.reference_scheme, dtype=np.int64)
    if instance.reference_allocation is not None:
        payload["reference_allocation_AUDIT_ONLY"] = np.asarray(instance.reference_allocation, dtype=np.int64)
    if instance.scheme_resource is not None:
        payload["scheme_resource_AUDIT_ONLY"] = np.asarray(instance.scheme_resource)
    np.savez_compressed(path, **payload)
    return path


def scalar_metrics(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v)
    return out


def bootstrap_ci(values: Iterable[float], *, seed: int = 2026, samples: int = 10000):
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    block = 500
    pos = 0
    while pos < samples:
        n = min(block, samples - pos)
        inds = rng.integers(0, x.size, size=(n, x.size))
        boot[pos:pos+n] = np.mean(x[inds], axis=1)
        pos += n
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mean, float(lo), float(hi)


def exact_mcnemar_p(a: int, b: int) -> float:
    n = int(a) + int(b)
    if n == 0:
        return 1.0
    k = min(int(a), int(b))
    lower = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * lower)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def read_json(path: Path, default=None):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return default
