#!/usr/bin/env python3
"""Build DeepWTA-Standard-only tables and integrity notes from per-case rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD = "DeepWTA-Standard"
METRICS = {
    "profile_score": "J",
    "ammo_ratio_to_raw_lb": "Ammo/LB",
    "crossing_pair_rate": "PCR",
    "avg_distance": "AvgDist",
    "avg_interception": "AvgIntercept",
}


def bootstrap_mean_ci(
    values: pd.Series, *, samples: int = 10_000, seed: int = 2026
) -> tuple[float, float, float]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(array))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=float)
    for start in range(0, samples, 500):
        stop = min(start + 500, samples)
        indexes = rng.integers(0, array.size, size=(stop - start, array.size))
        bootstrap[start:stop] = np.mean(array[indexes], axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return mean, float(low), float(high)


def validate_rows(frame: pd.DataFrame, *, min_cases: int) -> None:
    required = {
        "scenario_index", "scenario_id", "method", "hard_feasible",
        "profile_score", "ammo_ratio_to_raw_lb", "crossing_pair_rate",
        "avg_distance", "avg_interception", "total_runtime_ms",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing result columns: {missing}")
    methods = set(frame["method"].dropna().astype(str))
    if methods != {METHOD}:
        raise ValueError(f"expected only {METHOD!r}, found {sorted(methods)}")
    if len(frame) < min_cases:
        raise ValueError(f"expected at least {min_cases} rows, found {len(frame)}")
    if frame["scenario_id"].duplicated().any():
        raise ValueError("duplicate scenario_id values in Standard result rows")


def summary_row(
    frame: pd.DataFrame, *, bootstrap_samples: int, seed: int
) -> dict[str, float | int | str]:
    feasible = pd.to_numeric(frame["hard_feasible"], errors="coerce").fillna(0.0)
    feasible_rows = frame.loc[feasible >= 1.0]
    row: dict[str, float | int | str] = {
        "Method": METHOD,
        "N": int(len(frame)),
        "Feasible": float(feasible.mean()),
    }
    for offset, (column, label) in enumerate(METRICS.items()):
        source = feasible_rows[column] if column == "profile_score" else frame[column]
        mean, low, high = bootstrap_mean_ci(
            source, samples=bootstrap_samples, seed=seed + offset
        )
        row[label] = mean
        row[f"{label}_CI95_Low"] = low
        row[f"{label}_CI95_High"] = high
    runtime = pd.to_numeric(frame["total_runtime_ms"], errors="coerce")
    row["RuntimeMeanMs"] = float(runtime.mean())
    row["RuntimeMedianMs"] = float(runtime.median())
    row["RuntimeP95Ms"] = float(runtime.quantile(0.95))
    return row


def group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouping = [column for column in ("group_id",) if column in frame.columns]
    groups = frame.groupby(grouping, dropna=False, sort=True) if grouping else [("all", frame)]
    rows = []
    for key, group in groups:
        label = key if isinstance(key, str) else key[0]
        rows.append({
            "Group": label,
            "N": int(len(group)),
            "Feasible": float(pd.to_numeric(group["hard_feasible"], errors="coerce").mean()),
            "J": float(pd.to_numeric(group["profile_score"], errors="coerce").mean()),
            "Ammo/LB": float(pd.to_numeric(group["ammo_ratio_to_raw_lb"], errors="coerce").mean()),
            "PCR": float(pd.to_numeric(group["crossing_pair_rate"], errors="coerce").mean()),
            "AvgDist": float(pd.to_numeric(group["avg_distance"], errors="coerce").mean()),
            "AvgIntercept": float(pd.to_numeric(group["avg_interception"], errors="coerce").mean()),
            "RuntimeMedianMs": float(pd.to_numeric(group["total_runtime_ms"], errors="coerce").median()),
        })
    return pd.DataFrame(rows)


def solution_inventory(solution_dir: Path | None) -> dict[str, int | str | None]:
    if solution_dir is None:
        return {"path": None, "files": 0, "nonempty_files": 0}
    files = sorted(solution_dir.glob("*.npz")) if solution_dir.is_dir() else []
    return {
        "path": solution_dir.as_posix(),
        "files": len(files),
        "nonempty_files": sum(path.stat().st_size > 0 for path in files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--solution-dir", type=Path, default=None)
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    frame = pd.read_csv(args.details)
    validate_rows(frame, min_cases=args.min_cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([
        summary_row(frame, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    ])
    groups = group_summary(frame)
    inventory = solution_inventory(args.solution_dir)
    table.to_csv(args.output_dir / "table_main.csv", index=False)
    groups.to_csv(args.output_dir / "group_summary.csv", index=False)
    frame.to_csv(args.output_dir / "all_result_rows.csv", index=False)
    payload = {
        "method": METHOD,
        "details": args.details.as_posix(),
        "rows": int(len(frame)),
        "unique_scenarios": int(frame["scenario_id"].nunique()),
        "solution_inventory": inventory,
    }
    (args.output_dir / "result_inventory.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    solution_note = (
        f"{inventory['nonempty_files']} non-empty of {inventory['files']} files"
        if inventory["path"] else "not checked"
    )
    report = (
        "# DeepWTA-Standard result summary\n\n"
        f"- Per-case rows: {len(frame)}\n"
        f"- Unique scenarios: {frame['scenario_id'].nunique()}\n"
        f"- Hard-feasibility rate: {table.loc[0, 'Feasible']:.6f}\n"
        f"- Decision archive: {solution_note}\n\n"
        "No cross-method comparisons or paired hypothesis tests are included.\n"
    )
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
