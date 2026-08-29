"""指标汇总工具。

evaluate_pointer_net_ac.py 与 evaluate_multi_head_dqn.py 中的 ``summarize``
实现完全一致（对行列表按指标列求 mean/std/median），故抽取为公共函数。
"""
from __future__ import annotations

import numpy as np


def summarize_rows(rows: list[dict], metrics: list[str]) -> dict:
    """对行列表按指定指标列计算 mean/std/median。

    Parameters
    ----------
    rows:
        每行为一个 dict，键为指标名。
    metrics:
        需要汇总的指标键列表。

    Returns
    -------
    dict:
        {"count": int, "metrics": {metric: {"mean","std","median"}}}
    """
    out = {"count": len(rows), "metrics": {}}
    for key in metrics:
        x = np.asarray([float(r[key]) for r in rows], dtype=float)
        out["metrics"][key] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "median": float(np.median(x)),
        }
    return out
