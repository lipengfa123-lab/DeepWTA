"""张量/数组转换与图物理量提取工具。

多个评估脚本重复实现了 ``to_np``（tensor->numpy）与 ``extract_physical``
（从 PyG 图提取物理量字典），故集中到本模块统一维护。
"""
from __future__ import annotations

import numpy as np


def to_np(value) -> np.ndarray:
    """把 torch.Tensor 或任意可数组化的值转为 numpy.ndarray。"""
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def tensor_np(value) -> np.ndarray:
    """等价于 ``to_np``，保留旧脚本中的命名。"""
    return to_np(value)


def extract_physical(graph) -> dict[str, np.ndarray]:
    """从 PyG 图对象提取下游评估所需的物理量字典（全部 float64）。"""
    return {
        "B": to_np(graph.B).astype(np.float64),
        "U": to_np(graph.U).astype(np.float64),
        "G": to_np(graph.G).astype(np.float64),
        "G_th": to_np(graph.G_th).astype(np.float64),
        "Q": to_np(graph.Q).astype(np.float64),
        "X": to_np(graph.X).astype(np.float64),
        "D": to_np(graph.D).astype(np.float64),
        "base_coords": to_np(graph.base_coords).astype(np.float64),
        "target_coords": to_np(graph.target_coords).astype(np.float64),
    }


def graph_meta_value(value, default):
    """把图元数据标量转为 python 标量；空值返回 default。"""
    if value is None:
        return default
    try:
        import torch

        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
    except Exception:
        pass
    return value
