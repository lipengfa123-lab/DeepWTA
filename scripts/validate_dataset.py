#!/usr/bin/env python3
"""Validate the exported TEST-800 NPZ contract without regenerating data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED = {
    "base_coords",
    "target_coords",
    "B_inventory",
    "U_scheme_demand",
    "G_damage_level",
    "G_th_damage_threshold",
    "Q_intercept_prob",
    "D_distance",
    "X_reachable",
    "scenario_index",
    "scenario_id",
    "split",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_case(path: Path) -> tuple[int, str]:
    if path.stat().st_size == 0:
        raise ValueError("empty archive")
    with np.load(path, allow_pickle=False) as case:
        missing = sorted(REQUIRED.difference(case.files))
        if missing:
            raise ValueError(f"missing keys: {missing}")
        index = int(case["scenario_index"])
        scenario_id = str(case["scenario_id"])
        if str(case["split"]) != "test":
            raise ValueError("split is not 'test'")
        q = case["Q_intercept_prob"]
        if q.ndim != 3:
            raise ValueError("Q_intercept_prob must have shape [K,N,M]")
        k, n, m = q.shape
        l = case["U_scheme_demand"].shape[0]
        expected = {
            "base_coords": (n, 2),
            "target_coords": (m, 2),
            "B_inventory": (k, n),
            "U_scheme_demand": (l, k, m),
            "G_damage_level": (l, m),
            "G_th_damage_threshold": (m,),
            "D_distance": (n, m),
            "X_reachable": (k, n, m),
        }
        bad = {
            name: (case[name].shape, shape)
            for name, shape in expected.items()
            if case[name].shape != shape
        }
        if bad:
            raise ValueError(f"shape mismatch: {bad}")
        if k != 10 or l != 10 or not (8 <= n <= 80 and 8 <= m <= 80):
            raise ValueError(f"outside frozen contract: K={k}, L={l}, N={n}, M={m}")
    return index, scenario_id


def verify_manifest(release_root: Path) -> int:
    manifest = release_root / "SHA256_MANIFEST.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    checked = 0
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rel = row.get("path", "")
            if not rel.startswith("inputs/cases/"):
                continue
            path = release_root / Path(rel)
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_size = int(row.get("size_bytes", row.get("bytes", -1)))
            if path.stat().st_size != expected_size:
                raise ValueError(f"size mismatch: {rel}")
            expected_hash = row.get("sha256", "").lower()
            if expected_hash and sha256(path) != expected_hash:
                raise ValueError(f"SHA-256 mismatch: {rel}")
            checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("results/reviewer2_release/test800_final"),
    )
    parser.add_argument("--expected-count", type=int, default=800)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()

    cases_dir = args.release_root / "inputs" / "cases"
    files = sorted(cases_dir.glob("idx_*.npz"))
    if len(files) != args.expected_count:
        raise RuntimeError(f"expected {args.expected_count} cases, found {len(files)}")

    indexes: set[int] = set()
    ids: set[str] = set()
    for path in files:
        try:
            index, scenario_id = check_case(path)
        except Exception as exc:
            raise RuntimeError(f"invalid case {path}: {exc}") from exc
        if index in indexes or scenario_id in ids:
            raise RuntimeError(f"duplicate index/id in {path}")
        indexes.add(index)
        ids.add(scenario_id)

    expected_indexes = set(range(args.expected_count))
    if indexes != expected_indexes:
        raise RuntimeError("scenario indexes are not exactly 0..expected_count-1")

    hash_count = 0 if args.skip_hashes else verify_manifest(args.release_root)
    summary = {
        "status": "PASS",
        "case_count": len(files),
        "unique_scenario_ids": len(ids),
        "manifest_input_hashes_checked": hash_count,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
