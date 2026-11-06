from __future__ import annotations

import importlib.util
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
    # The dataset release ships rendered figures, but not the full project.
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    missing_modules = []
    for name in ("deepwta.data.manifest", "deepwta.evaluation.exact_metrics",
                 "deepwta.evaluation.profile_objective"):
        try:
            available = importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            available = False
        if not available:
            missing_modules.append(name)
    if missing_modules:
        figures = [FIGURE_DIR / f"000149_{method}_dark.{extension}"
                   for _, method in PANELS for extension in ("png", "pdf")]
        missing_files = [str(p) for p in figures if not p.is_file() or p.stat().st_size == 0]
        if missing_files:
            raise FileNotFoundError(
                "Full-project modules are missing: " + ", ".join(missing_modules)
                + ". Released figure files are also missing or empty: "
                + ", ".join(missing_files)
                + ". Restore results/figures/fig6_case_study from the dataset release."
            )
        print("[fig6] Full-project modules unavailable: " + ", ".join(missing_modules))
        print("[fig6] Reusing 6 released PNG/PDF panels; no figures were regenerated.")
        print(f"[fig6] Output: {FIGURE_DIR}")
        return
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
