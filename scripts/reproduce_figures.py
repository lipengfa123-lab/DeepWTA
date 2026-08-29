#!/usr/bin/env python3
"""Generate DeepWTA-Standard TEST-800 metric distributions."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOTS = (
    ("profile_score", "Canonical objective J", "objective_J"),
    ("ammo_ratio_to_raw_lb", "Ammo / lower bound", "ammo_ratio"),
    ("crossing_pair_rate", "Pairwise crossing rate", "PCR"),
    ("total_runtime_ms", "Total runtime (ms)", "runtime"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details", type=Path,
        default=Path("results/reviewer2_release/test800_final/main/deepwta/test_details.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reproduced_results/figures")
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.details)
    if set(frame["method"].astype(str)) != {"DeepWTA-Standard"}:
        raise ValueError("details must contain only DeepWTA-Standard rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for column, xlabel, stem in PLOTS:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.hist(values, bins=30, color="#2c6eaa", edgecolor="white")
        ax.axvline(values.median(), color="#b23a48", linestyle="--", label="median")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Scenario count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{stem}.png", dpi=240)
        fig.savefig(args.output_dir / f"{stem}.pdf")
        plt.close(fig)
    print(f"figures: {args.output_dir}")


if __name__ == "__main__":
    main()
