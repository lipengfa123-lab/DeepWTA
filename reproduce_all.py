from __future__ import annotations

"""DeepWTA Dataset 一键式复现脚本。

运行保留的论文图（仅 Fig.6；Fig.3 / Fig.4 / Fig.5 / Fig.7 均已从本数据集移除）
以及从聚合结果生成 Table 4。

用法：
    python reproduce_all.py                 # 制表 + 绘图
    python reproduce_all.py --skip-tables   # 只绘图
    python reproduce_all.py --skip-figures  # 只制表
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 仅保留 Fig.6；Fig.3（战场图）、Fig.4、Fig.5 与 Fig.7（消融）均已移除。
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
    # Fig.6 依赖 data/splits 的 manifest 与 dataset/fig6_case_study 的输入/决策数据。
    manifest = ROOT / "data" / "splits" / "wta_e2e_v1.json"
    case_input = ROOT / "dataset" / "fig6_case_study" / "inputs" / "idx_000149.npz"
    return manifest.is_file() and case_input.is_file()


def figures() -> None:
    for rel in FIG_SCRIPTS:
        script = ROOT / rel
        if not script.is_file():
            print(f"[skip] 缺少脚本: {rel}")
            continue
        if rel.endswith("fig6_case_study.py") and not _fig6_ready():
            print("[skip] Fig.6 缺少 data/splits 或 dataset/fig6_case_study 数据（见 DeepWTA_master 完整仓库）。")
            continue
        _run(rel)


def tables() -> None:
    script = ROOT / "scripts" / "experiments" / "build_table4_scaling.py"
    if not script.is_file():
        print("[skip] 缺少 build_table4_scaling.py")
        return
    _run("scripts/experiments/build_table4_scaling.py")


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepWTA Dataset 一键式复现（绘图 + 制表）。")
    ap.add_argument("--skip-figures", action="store_true", help="跳过绘图")
    ap.add_argument("--skip-tables", action="store_true", help="跳过制表")
    args = ap.parse_args()

    if not args.skip_tables:
        tables()
    if not args.skip_figures:
        figures()

    print("\nDone. 图见 results/figures/。")


if __name__ == "__main__":
    main()
