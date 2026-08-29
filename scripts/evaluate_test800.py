#!/usr/bin/env python3
"""Dispatch DeepWTA-Standard on the frozen TEST-800 split."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


MANIFEST = Path("data/splits/wta_e2e_v1.json")
CHECKPOINT = Path(
    "checkpoints/deepwta/e2e_deepwta_wta_e2e_v1_seed2026/best_full_feasible.pt"
)


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.reviewer2.evaluate_deepwta_release",
        "--checkpoint",
        str(CHECKPOINT),
        "--dataset-manifest",
        str(MANIFEST),
        "--split",
        "test",
        "--limit",
        str(args.limit),
        "--device",
        args.device,
        "--output-dir",
        str(args.output_root / "deepwta"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=1,
        help="1 is a safe smoke test; 0 means all 800 cases",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root", type=Path, default=Path("reproduced_results/test800")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    command = build_command(args)
    print(shlex.join(command))
    if args.dry_run:
        return
    missing = [str(path) for path in (MANIFEST, CHECKPOINT) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files: " + ", ".join(missing))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
