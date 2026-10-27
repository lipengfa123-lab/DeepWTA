from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "fig6_case_study"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "fig6_case_study"
MANIFEST = PROJECT_ROOT / "data" / "splits" / "wta_e2e_v1.json"
RENDERER = PROJECT_ROOT / "scripts" / "visualization" / "plot_saved_wta_solution_map.py"

PANELS = [
    ("a", "DeepWTA-Standard"),
    ("b", "MultiHead-DQN-E2E"),
    ("c", "PointerNet-AC-E2E"),
    ("d", "FRA-DC"),
    ("e", "SA-MC"),
    ("f", "GA-MC"),
]

INPUT_NPZ = DATA_DIR / "inputs" / "idx_000149.npz"


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for panel, method in PANELS:
        solution = DATA_DIR / "solutions" / method / "idx_000149.npz"
        print(f"[fig6] ({panel}) {method}")
        subprocess.run(
            [
                sys.executable, str(RENDERER),
                "--dataset-manifest", str(MANIFEST),
                "--input-npz", str(INPUT_NPZ),
                "--solution-npz", str(solution),
                "--method-label", method,
                "--output-dir", str(FIGURE_DIR),
                "--style", "dark",
                "--formats", "png,pdf",
                "--dpi", "320",
                "--allow-infeasible",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
