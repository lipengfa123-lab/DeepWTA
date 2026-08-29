from __future__ import annotations

from dataclasses import dataclass

from configs.dataset_config import CONFIG as DATA_CONFIG


@dataclass(frozen=True)
class DeepWTAModelConfig:
    num_weapons: int = DATA_CONFIG.wta.num_weapons
    num_schemes: int = DATA_CONFIG.wta.num_schemes

    # Must match v5 graph construction.
    node_dim: int = (
        4
        + DATA_CONFIG.wta.num_weapons
        + DATA_CONFIG.wta.num_schemes * DATA_CONFIG.wta.num_weapons
        + DATA_CONFIG.wta.num_schemes
    )
    edge_dim: int = 1 + 2 * DATA_CONFIG.wta.num_weapons

    inventory_max: float = float(DATA_CONFIG.wta.inventory_max)
    demand_max: float = float(DATA_CONFIG.wta.demand_max)

    hidden_dim: int = 96
    heads: int = 4
    num_encoder_layers: int = 3

    scheme_dim: int = 64
    scheme_hidden_dim: int = 128
    edge_mlp_dim: int = 256

    dropout: float = 0.10
    survival_floor: float = 0.05

    # Enforce B exactly in the continuous forward pass. Demand is then handled by
    # training losses and, finally, tactical repair after discretization.
    use_inventory_projection: bool = True

    # Phase-1 neural-side ablation. Disabled by default so the formal T2b
    # checkpoint retains its exact architecture and numerical behavior.
    use_scheme_efficiency_features: bool = False

    # Phase-2 assignment-logit ablation. ``none`` is the formal B0 baseline.
    tactical_assignment_bias: str = "none"
    tactical_bias_alpha_init: float = 0.05

    # Phase-3A inference/training ablation. One preserves the formal B0 path.
    assignment_temperature: float = 1.0

    # Phase-6B-1 inference-only ablation. False preserves the formal B0 path.
    zero_inventory_assignment_mask: bool = False

    # Phase-6C scheme-side ablation. False preserves the damage-only mask.
    resource_feasible_scheme_mask: bool = False

    # Phase-6B-2R one-step positive-stock contention correction. Zero is off.
    positive_inventory_contention_lambda: float = 0.0

    # Phase-3B residual-inventory redistribution. Disabled for formal B0.
    redistribution_mode: str = "none"
    redistribution_beta: float = 0.0
    redistribution_iterations: int = 2
    redistribution_eta: float = 0.75


MODEL_CONFIG = DeepWTAModelConfig()
