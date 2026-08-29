#!/usr/bin/env python3
"""Create an audited canonical CSV by replacing explicitly rerun rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from deepwta.evaluation.metric_schema import METRIC_SCHEMA_COLUMNS

KEYS = ["method", "scenario_id", "dataset_name"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--patch", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = pd.read_csv(args.base)
    if tuple(base.columns) != tuple(METRIC_SCHEMA_COLUMNS):
        raise ValueError("base metric schema mismatch")
    if base.duplicated(KEYS).any():
        raise ValueError("base contains duplicate metric rows")

    replacement = pd.concat([pd.read_csv(path) for path in args.patch], ignore_index=True)
    if tuple(replacement.columns) != tuple(METRIC_SCHEMA_COLUMNS):
        raise ValueError("patch metric schema mismatch")
    if replacement.duplicated(KEYS).any():
        raise ValueError("patches contain duplicate metric rows")
    base_keys = set(map(tuple, base[KEYS].itertuples(index=False, name=None)))
    patch_keys = set(map(tuple, replacement[KEYS].itertuples(index=False, name=None)))
    missing = patch_keys - base_keys
    if missing:
        raise ValueError(f"patch rows absent from base: {sorted(missing)[:3]}")
    if (pd.to_numeric(replacement["hard_feasible"], errors="coerce") < 1.0).any():
        raise ValueError("a replacement row is still hard infeasible")

    base_indexed = base.set_index(KEYS)
    replacement_indexed = replacement.set_index(KEYS)
    base_indexed.loc[replacement_indexed.index, :] = replacement_indexed
    merged = base_indexed.reset_index()[list(METRIC_SCHEMA_COLUMNS)]
    if len(merged) != len(base) or merged.duplicated(KEYS).any():
        raise RuntimeError("merged metric cardinality changed")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    audit = {
        "base": str(args.base.resolve()),
        "patches": [str(path.resolve()) for path in args.patch],
        "output": str(output),
        "rows": int(len(merged)),
        "replaced_rows": int(len(replacement)),
        "all_hard_feasible": bool((merged["hard_feasible"] >= 1.0).all()),
    }
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"metrics: {output}")
    print(f"audit  : {audit_path}")


if __name__ == "__main__":
    main()
