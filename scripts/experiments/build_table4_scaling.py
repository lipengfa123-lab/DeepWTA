#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "scaling" / "analysis" / "scaling_summary_large_xlarge.csv"

SCALE_ORDER = ["LARGE", "XLARGE"]
METHOD_ORDER = [
    "DeepWTA-Standard",
    "MultiHead-DQN-E2E",
    "PointerNet-AC-E2E",
    "FRA-DC",
    "SA-MC",
    "GA-MC",
]
LABEL = {
    "DeepWTA-Standard": "DeepWTA (Proposed)",
    "MultiHead-DQN-E2E": "MultiHead-DQN",
    "PointerNet-AC-E2E": "PointerNet-AC",
    "FRA-DC": "FRA-DC",
    "SA-MC": "SA-MC",
    "GA-MC": "GA-MC",
}


def number(value: str | None) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x


def fmt(value: float, digits: int = 2) -> str:
    return "—" if not math.isfinite(value) else f"{value:.{digits}f}"


def pct(fraction: float) -> str:
    if not math.isfinite(fraction):
        return "—"
    value = Decimal(str(100.0 * fraction)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{value:.2f}"


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build(rows: list[dict]) -> list[dict]:
    index = {(r["Scale"], r["Method"]): r for r in rows}
    out = []
    for scale in SCALE_ORDER:
        for method in METHOD_ORDER:
            r = index.get((scale, method))
            if r is None:
                continue
            completed = number(r.get("CompletedCases"))
            success = number(r.get("EndToEndSuccessRate"))
            timeout = "Yes" if completed == 0 else "No"
            out.append({
                "Scale": scale,
                "Method": LABEL[method],
                "Timeout": timeout,
                "Feasible/%": "0.00" if timeout == "Yes" else pct(success),
                "Obj. J": fmt(number(r.get("J_FeasibleMean"))),
                "WR": fmt(number(r.get("WR"))),
                "TCR": fmt(number(r.get("TCR"))),
                "Timeliness": fmt(number(r.get("Timeliness"))),
                "Inter. ratio": fmt(number(r.get("Inter. ratio"))),
                "Time/ms": fmt(number(r.get("RuntimeMedianMs_completed_only"))),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reproduced" / "tables")
    args = ap.parse_args()

    table = build(load(args.summary))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fields = list(table[0].keys())
    with (args.output_dir / "table4_large_xlarge.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join(["---"] * len(fields)) + "|",
    ]
    for row in table:
        lines.append("| " + " | ".join(str(row[f]) for f in fields) + " |")
    (args.output_dir / "table4_large_xlarge.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote: {args.output_dir / 'table4_large_xlarge.csv'}")


if __name__ == "__main__":
    main()
