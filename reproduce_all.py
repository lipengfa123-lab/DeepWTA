from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


FIG_SCRIPTS = [
    "scripts/figures/fig6_case_study.py",
]


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    return env


def _run(rel: str) -> None:
    print(f"[run] {rel}")
    subprocess.run([sys.executable, rel], cwd=ROOT, env=_env(), check=True)


def _fig6_ready() -> bool:
    manifest = ROOT / "data" / "splits" / "wta_e2e_v1.json"
    case_input = ROOT / "fig6_case_study" / "inputs" / "idx_000149.npz"
    return manifest.is_file() and case_input.is_file()


def figures() -> None:
    for rel in FIG_SCRIPTS:
        script = ROOT / rel
        if not script.is_file():
            print(f"[skip] missing script: {rel}")
            continue
        if rel.endswith("fig6_case_study.py") and not _fig6_ready():
            print("[skip] Fig.6 missing data/splits or fig6_case_study data (see DeepWTA_master full repository).")
            continue
        _run(rel)


def tables() -> None:
    script = ROOT / "scripts" / "experiments" / "build_table4_scaling.py"
    if not script.is_file():
        print("[skip] missing build_table4_scaling.py")
        return
    _run("scripts/experiments/build_table4_scaling.py")


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepWTA Dataset one-click reproduction (tables + figures).")
    ap.add_argument("--skip-figures", action="store_true", help="skip figures")
    ap.add_argument("--skip-tables", action="store_true", help="skip tables")
    args = ap.parse_args()

    if not args.skip_tables:
        tables()
    if not args.skip_figures:
        figures()

    print("\nDone. See results/figures/.")


if __name__ == "__main__":
    main()
