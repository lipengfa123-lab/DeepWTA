#!/usr/bin/env python3
"""Rebuild the DeepWTA-Standard TEST-800 summary table."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details", type=Path,
        default=Path("results/reviewer2_release/test800_final/main/deepwta/test_details.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reproduced_results/main_table")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable, "-m", "scripts.reviewer2.summarize_standard_results",
            "--details", str(args.details), "--output-dir", str(args.output_dir),
            "--min-cases", "800", "--bootstrap-samples", str(args.bootstrap_samples),
            "--seed", str(args.seed),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
