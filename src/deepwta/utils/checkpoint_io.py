"""DeepWTA 主模型 checkpoint 的加载与构建工具。

多个 evaluate / benchmark 脚本都需要"从 checkpoint 重建 DeepWTAModel 并加载
权重"这一逻辑，且实现完全一致，故抽取为本模块。
"""
from __future__ import annotations

import torch

from deepwta.models.deepwta_model import DeepWTAModel


def sync_device(device: torch.device) -> None:
    """在 CUDA 上同步设备，保证计时/指标统计不受异步执行影响。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_checkpoint(path, device: torch.device) -> dict:
    """加载 DeepWTA checkpoint 字典（map_location 到指定设备）。"""
    return torch.load(path, map_location=device, weights_only=False)


def build_deepwta_model(checkpoint: dict, device: torch.device) -> DeepWTAModel:
    """从 checkpoint 重建 DeepWTAModel 并加载权重。

    checkpoint 需包含 ``model_config``（模型结构参数）与 ``model_state``
    （状态字典）。模型以 eval 模式返回。
    """
    c = checkpoint["model_config"]
    model = DeepWTAModel(
        node_dim=int(c["node_dim"]),
        edge_dim=int(c["edge_dim"]),
        num_weapons=int(c["num_weapons"]),
        num_schemes=int(c["num_schemes"]),
        inventory_max=float(c["inventory_max"]),
        demand_max=float(c["demand_max"]),
        hidden_dim=int(c["hidden_dim"]),
        heads=int(c["heads"]),
        num_encoder_layers=int(c["num_encoder_layers"]),
        scheme_dim=int(c["scheme_dim"]),
        scheme_hidden_dim=int(c["scheme_hidden_dim"]),
        edge_mlp_dim=int(c["edge_mlp_dim"]),
        dropout=float(c["dropout"]),
        survival_floor=float(c["survival_floor"]),
        use_inventory_projection=bool(c["use_inventory_projection"]),
        use_scheme_efficiency_features=bool(
            c.get("use_scheme_efficiency_features", False)
        ),
        tactical_assignment_bias=str(
            c.get("tactical_assignment_bias", "none")
        ),
        tactical_bias_alpha_init=float(
            c.get("tactical_bias_alpha_init", 0.05)
        ),
        assignment_temperature=float(
            c.get("assignment_temperature", 1.0)
        ),
        zero_inventory_assignment_mask=bool(
            c.get("zero_inventory_assignment_mask", False)
        ),
        resource_feasible_scheme_mask=bool(
            c.get("resource_feasible_scheme_mask", False)
        ),
        positive_inventory_contention_lambda=float(
            c.get("positive_inventory_contention_lambda", 0.0)
        ),
        redistribution_mode=str(c.get("redistribution_mode", "none")),
        redistribution_beta=float(c.get("redistribution_beta", 0.0)),
        redistribution_iterations=int(
            c.get("redistribution_iterations", 2)
        ),
        redistribution_eta=float(c.get("redistribution_eta", 0.75)),
        model_variant=str(c.get("model_variant", "legacy")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
