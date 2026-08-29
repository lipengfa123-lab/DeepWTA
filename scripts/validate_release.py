#!/usr/bin/env python3
"""Read-only integrity check for the retained DeepWTA-Standard archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("results/reviewer2_release/test800_final"),
    )
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--expected-inputs", type=int, default=800)
    parser.add_argument("--expected-solutions", type=int, default=0)
    args = parser.parse_args()

    manifest_path = args.release_root / "SHA256_MANIFEST.csv"
    failures: list[str] = []
    checked = 0
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        rel = row["path"]
        path = args.release_root / Path(rel)
        if not path.is_file():
            failures.append(f"missing: {rel}")
            continue
        expected_size = int(row.get("size_bytes", row.get("bytes", -1)))
        if path.stat().st_size != expected_size:
            failures.append(
                f"size: {rel} expected={expected_size} actual={path.stat().st_size}"
            )
            continue
        if not args.skip_hashes and row.get("sha256"):
            if sha256(path) != row["sha256"].lower():
                failures.append(f"sha256: {rel}")
                continue
        checked += 1

    inputs = sorted((args.release_root / "inputs" / "cases").glob("*.npz"))
    solutions = sorted(
        (args.release_root / "main" / "deepwta" / "solutions" / "DeepWTA-Standard").glob("*.npz")
    )
    nonempty_solutions = sum(path.stat().st_size > 0 for path in solutions)
    if len(inputs) != args.expected_inputs:
        failures.append(f"input count: expected={args.expected_inputs} actual={len(inputs)}")
    if len(solutions) != args.expected_solutions:
        failures.append(
            f"solution count: expected={args.expected_solutions} actual={len(solutions)}"
        )
    if nonempty_solutions != args.expected_solutions:
        failures.append(
            "non-empty solution count: "
            f"expected={args.expected_solutions} actual={nonempty_solutions}"
        )

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "manifest_rows": len(rows),
        "manifest_rows_verified": checked,
        "input_files": len(inputs),
        "solution_files": len(solutions),
        "nonempty_solution_files": nonempty_solutions,
        "failures": failures[:50],
        "failure_count": len(failures),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
