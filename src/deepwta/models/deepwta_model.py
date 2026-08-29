from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter, softmax

from deepwta.models.graph_encoder import ResidualGraphAttentionBlock


SchemeMode = Literal["soft", "gumbel", "hard"]
TacticalBiasMode = Literal["none", "distance", "interception", "both"]
RedistributionMode = Literal["none", "capacity", "network"]


@dataclass
class DeepWTAOutput:
    """Continuous outputs produced by DeepWTAModel.

    All allocations remain continuous. Integer rounding / tactical repair must be
    performed outside the model.
    """

    allocation: Tensor                 # [E, K], after inventory projection
    allocation_pre_projection: Tensor  # [E, K]
    assignment_probs: Tensor           # [E, K], normalized over bases per target
    assignment_logits: Tensor          # [E, K]
    scheme_probs: Tensor               # [T, L]
    scheme_logits: Tensor              # [T, L]
    target_demand: Tensor              # [T, K], scheme-weighted physical demand
    route_load: Tensor                 # [E], sum_k allocation[e, k]
    target_node_index: Tensor          # [T], global node ids in the PyG batch
    edge_target_index: Tensor          # [E], compact 0..T-1 target ids
    inventory_scale: Tensor            # [num_nodes, K]
    scheme_efficiency_features: Tensor | None = None  # [T, L, 5]
    assignment_net_logits: Tensor | None = None        # [E, K]
    assignment_tactical_bias: Tensor | None = None     # [E, K]
    redistribution_diagnostics: dict[str, Tensor] | None = None


class DeepWTAModel(nn.Module):
    """Universal two-stage GNN for variable-size WTA instances.

    Expected v5 node layout:
        [x_norm, y_norm, node_type,
         B_1..B_K,
         flatten(U_{1..L,1..K}),
         G_1..G_L,
         G_th]

    Expected edge layout:
        [D_norm, Q_1..Q_K, X_1..X_K]

    The original WTA graph stores only base -> target edges. The encoder creates
    reverse target -> base edges internally so bases can aggregate information
    from competing targets. Assignment decoding still uses only the original
    base -> target routes.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_weapons: int,
        num_schemes: int,
        inventory_max: float,
        demand_max: float,
        hidden_dim: int = 96,
        heads: int = 4,
        num_encoder_layers: int = 3,
        scheme_dim: int = 64,
        scheme_hidden_dim: int = 128,
        edge_mlp_dim: int = 256,
        dropout: float = 0.10,
        survival_floor: float = 0.05,
        use_inventory_projection: bool = True,
        use_scheme_efficiency_features: bool = False,
        tactical_assignment_bias: TacticalBiasMode = "none",
        tactical_bias_alpha_init: float = 0.05,
        assignment_temperature: float = 1.0,
        zero_inventory_assignment_mask: bool = False,
        resource_feasible_scheme_mask: bool = False,
        positive_inventory_contention_lambda: float = 0.0,
        redistribution_mode: RedistributionMode = "none",
        redistribution_beta: float = 0.0,
        redistribution_iterations: int = 2,
        redistribution_eta: float = 0.75,
        model_variant: str = "legacy",
    ):
        super().__init__()

        if num_weapons <= 0 or num_schemes <= 0:
            raise ValueError("num_weapons and num_schemes must be positive")
        if inventory_max <= 0 or demand_max <= 0:
            raise ValueError("inventory_max and demand_max must be positive")
        if num_encoder_layers < 1:
            raise ValueError("num_encoder_layers must be at least 1")

        expected_node_dim = 4 + num_weapons + num_schemes * num_weapons + num_schemes
        expected_edge_dim = 1 + 2 * num_weapons
        if node_dim != expected_node_dim:
            raise ValueError(
                f"node_dim={node_dim}, expected {expected_node_dim} "
                f"for K={num_weapons}, L={num_schemes}"
            )
        if edge_dim != expected_edge_dim:
            raise ValueError(
                f"edge_dim={edge_dim}, expected {expected_edge_dim} "
                f"for K={num_weapons}"
            )

        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.num_weapons = int(num_weapons)
        self.num_schemes = int(num_schemes)
        self.inventory_max = float(inventory_max)
        self.demand_max = float(demand_max)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.width = int(hidden_dim * heads)
        self.scheme_dim = int(scheme_dim)
        self.survival_floor = float(survival_floor)
        self.use_inventory_projection = bool(use_inventory_projection)
        self.use_scheme_efficiency_features = bool(
            use_scheme_efficiency_features
        )
        if tactical_assignment_bias not in {
            "none", "distance", "interception", "both"
        }:
            raise ValueError(
                "tactical_assignment_bias must be one of "
                "none/distance/interception/both"
            )
        if tactical_bias_alpha_init <= 0.0:
            raise ValueError("tactical_bias_alpha_init must be positive")
        self.tactical_assignment_bias = tactical_assignment_bias
        self.tactical_bias_alpha_init = float(tactical_bias_alpha_init)
        if not 0.0 < assignment_temperature <= 1.0:
            raise ValueError("assignment_temperature must be in (0,1]")
        self.assignment_temperature = float(assignment_temperature)
        self.zero_inventory_assignment_mask = bool(
            zero_inventory_assignment_mask
        )
        self.resource_feasible_scheme_mask = bool(
            resource_feasible_scheme_mask
        )
        if positive_inventory_contention_lambda < 0.0:
            raise ValueError("positive_inventory_contention_lambda must be nonnegative")
        self.positive_inventory_contention_lambda = float(
            positive_inventory_contention_lambda
        )
        if (
            self.positive_inventory_contention_lambda > 0.0
            and not self.zero_inventory_assignment_mask
        ):
            raise ValueError("contention correction requires zero-inventory assignment mask")
        if (
            self.positive_inventory_contention_lambda > 0.0
            and self.resource_feasible_scheme_mask
        ):
            raise ValueError("Phase-6B-2R must not use the Phase-6C scheme mask")
        if redistribution_mode not in {"none", "capacity", "network"}:
            raise ValueError("redistribution_mode must be none/capacity/network")
        if redistribution_beta < 0.0:
            raise ValueError("redistribution_beta must be nonnegative")
        if not 1 <= redistribution_iterations <= 3:
            raise ValueError("redistribution_iterations must be in [1,3]")
        if not 0.0 < redistribution_eta <= 1.0:
            raise ValueError("redistribution_eta must be in (0,1]")
        if redistribution_mode == "capacity" and redistribution_beta != 0.0:
            raise ValueError("capacity redistribution requires beta=0")
        if redistribution_mode != "none" and (
            self.use_scheme_efficiency_features
            or self.tactical_assignment_bias != "none"
            or self.assignment_temperature != 1.0
            or self.zero_inventory_assignment_mask
            or self.resource_feasible_scheme_mask
            or self.positive_inventory_contention_lambda > 0.0
        ):
            raise ValueError("Phase-3B redistribution requires formal B0 settings")
        self.redistribution_mode = redistribution_mode
        self.redistribution_beta = float(redistribution_beta)
        self.redistribution_iterations = int(redistribution_iterations)
        self.redistribution_eta = float(redistribution_eta)
        self.model_variant = str(model_variant)

        if self.model_variant == "context_v4":
            from deepwta.models.resource_context_encoder import ResourceContextEncoder
            from deepwta.models.cost_aware_bias import CostAwareBias
            self.resource_context_encoder = ResourceContextEncoder(128)
            self.cost_aware_bias = CostAwareBias()

        blocks = []
        in_dim = node_dim
        for _ in range(num_encoder_layers):
            blocks.append(
                ResidualGraphAttentionBlock(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
            )
            in_dim = self.width
        self.encoder = nn.ModuleList(blocks)

        # One descriptor is constructed for every (target, scheme):
        # [U_l1..U_lK, G_l, G_l - G_th].
        scheme_descriptor_dim = num_weapons + 2
        self.scheme_encoder = nn.Sequential(
            nn.Linear(scheme_descriptor_dim, scheme_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(scheme_hidden_dim, scheme_dim),
            nn.ELU(),
        )

        # Keep the checkpoint-era descriptor and encoder untouched. Phase-1
        # efficiency features enter through a zero-initialized residual branch,
        # allowing an old T2b checkpoint to be used as the exact B0 baseline.
        self.scheme_efficiency_dim = 5
        if self.use_scheme_efficiency_features:
            self.scheme_efficiency_encoder = nn.Sequential(
                nn.Linear(self.scheme_efficiency_dim, scheme_dim),
                nn.ELU(),
                nn.Linear(scheme_dim, scheme_dim),
            )
            nn.init.zeros_(self.scheme_efficiency_encoder[-1].weight)
            nn.init.zeros_(self.scheme_efficiency_encoder[-1].bias)

        # Score each damage scheme separately. This is preferable to producing L
        # unrelated logits from a single target embedding.
        self.scheme_scorer = nn.Sequential(
            nn.Linear(self.width + scheme_dim + (128 if self.model_variant == "context_v4" else 0), scheme_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(scheme_hidden_dim, 1),
        )

        # The edge decoder is explicitly conditioned on:
        #   - selected/relaxed scheme context;
        #   - scheme-weighted demand vector.
        spatial_dim = 3  # delta_x, delta_y, euclidean distance
        assignment_input_dim = (
            2 * self.width
            + edge_dim
            + spatial_dim
            + scheme_dim
            + num_weapons
        )
        self.assignment_decoder = nn.Sequential(
            nn.Linear(assignment_input_dim, edge_mlp_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_mlp_dim, edge_mlp_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_mlp_dim // 2, num_weapons),
        )
        raw_alpha_init = math.log(math.expm1(self.tactical_bias_alpha_init))
        if tactical_assignment_bias in {"distance", "both"}:
            self.tactical_distance_raw_alpha = nn.Parameter(
                torch.tensor(raw_alpha_init, dtype=torch.float32)
            )
        if tactical_assignment_bias in {"interception", "both"}:
            self.tactical_interception_raw_alpha = nn.Parameter(
                torch.tensor(raw_alpha_init, dtype=torch.float32)
            )

    # ------------------------------------------------------------------
    # Feature parsing
    # ------------------------------------------------------------------
    def _split_node_features(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        k = self.num_weapons
        l = self.num_schemes

        node_type = x[:, 2]
        base_mask = node_type < 0.5
        target_mask = ~base_mask

        inventory_norm = x[:, 3 : 3 + k]

        offset = 3 + k
        demand_flat = x[:, offset : offset + l * k]
        offset += l * k

        damage_norm = x[:, offset : offset + l]
        threshold_norm = x[:, -1]

        target_demand_norm = demand_flat[target_mask].reshape(-1, l, k)
        target_damage_norm = damage_norm[target_mask]
        target_threshold_norm = threshold_norm[target_mask]

        return (
            base_mask,
            target_mask,
            inventory_norm,
            target_demand_norm,
            target_damage_norm,
            target_threshold_norm,
        )

    def _validate_graph(self, data, base_mask: Tensor, target_mask: Tensor) -> None:
        if data.x.ndim != 2 or data.x.size(-1) != self.node_dim:
            raise ValueError(
                f"data.x must have shape [V,{self.node_dim}], got {tuple(data.x.shape)}"
            )
        if data.edge_attr.ndim != 2 or data.edge_attr.size(-1) != self.edge_dim:
            raise ValueError(
                f"data.edge_attr must have shape [E,{self.edge_dim}], "
                f"got {tuple(data.edge_attr.shape)}"
            )
        if data.edge_index.ndim != 2 or data.edge_index.size(0) != 2:
            raise ValueError("data.edge_index must have shape [2,E]")

        src, dst = data.edge_index
        if src.numel() == 0:
            raise ValueError("WTA graph contains no base-target edges")

        # v5 WTA graphs must store original assignment edges in base -> target direction.
        if not bool(torch.all(base_mask[src])):
            raise ValueError("WTA edge_index contains a non-base source node")
        if not bool(torch.all(target_mask[dst])):
            raise ValueError("WTA edge_index contains a non-target destination node")

    # ------------------------------------------------------------------
    # Topology encoder
    # ------------------------------------------------------------------
    @staticmethod
    def _bidirectional_edges(edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        reverse_index = edge_index.flip(0)
        return (
            torch.cat([edge_index, reverse_index], dim=1),
            torch.cat([edge_attr, edge_attr], dim=0),
        )

    def encode(self, data) -> Tensor:
        edge_index, edge_attr = self._bidirectional_edges(
            data.edge_index,
            data.edge_attr,
        )

        h = data.x
        for block in self.encoder:
            h = block(h, edge_index, edge_attr)
        return h


    def _resource_context(self, data):
        if self.model_variant != "context_v4":
            return None
        k=self.num_weapons
        inv=data.x[:,3:3+k]
        # aggregate base inventory pressure
        base_inv=inv[data.x[:,2]<0.5].mean(dim=0)
        target_dem=data.x[data.x[:,2]>=0.5,3+k:3+k+self.num_schemes*k]
        target_dem=target_dem.reshape(-1,self.num_schemes,k).mean(dim=(0,1))
        scarcity=target_dem/(base_inv.clamp_min(1e-6))
        accessibility=torch.ones_like(scarcity)*0.5
        pressure=1.0/(base_inv.clamp_min(1e-6))
        return self.resource_context_encoder(scarcity, accessibility, pressure)

    # ------------------------------------------------------------------
    # Stage 1: damage-scheme decoder
    # ------------------------------------------------------------------
    def _resource_feasible_schemes(
        self,
        data,
        target_mask: Tensor,
        inventory_norm: Tensor,
        target_demand_norm: Tensor,
    ) -> Tensor:
        """Return [T,L] availability feasibility without changing physical X."""
        k = self.num_weapons
        src, dst = data.edge_index
        edge_x = data.edge_attr[:, 1 + k : 1 + 2 * k] > 0.5
        inventory_available = inventory_norm[src] > 0.0
        available_by_node = scatter(
            (edge_x & inventory_available).to(target_demand_norm.dtype),
            dst,
            dim=0,
            dim_size=data.x.size(0),
            reduce="sum",
        )
        available = available_by_node[target_mask] > 0.0  # [T,K]
        return torch.all(
            (target_demand_norm <= 1e-8) | available.unsqueeze(1),
            dim=-1,
        )

    def _scheme_efficiency_features(
        self,
        data,
        target_mask: Tensor,
        target_demand_norm: Tensor,
    ) -> Tensor:
        """Return tensorized [raw, survival, reach, Q, distance] statistics."""
        k = self.num_weapons
        src, dst = data.edge_index
        del src  # Statistics aggregate over routes into each target.

        target_nodes = torch.nonzero(target_mask, as_tuple=False).view(-1)
        target_lookup = torch.full(
            (data.x.size(0),),
            -1,
            dtype=torch.long,
            device=data.x.device,
        )
        target_lookup[target_nodes] = torch.arange(
            target_nodes.numel(), device=data.x.device
        )
        edge_target = target_lookup[dst]
        if bool(torch.any(edge_target < 0)):
            raise ValueError("scheme efficiency edge points to a non-target")

        edge_attr = data.edge_attr
        distance = edge_attr[:, 0].clamp(0.0, 1.0)
        interception = edge_attr[:, 1 : 1 + k].clamp(0.0, 1.0)
        reachable = edge_attr[:, 1 + k : 1 + 2 * k].clamp(0.0, 1.0)
        target_count = target_nodes.numel()

        reach_count = scatter(
            reachable,
            edge_target,
            dim=0,
            dim_size=target_count,
            reduce="sum",
        )
        safe_count = reach_count.clamp_min(1.0)
        has_route = reach_count > 0.0

        survival_sum = scatter(
            (1.0 - interception) * reachable,
            edge_target,
            dim=0,
            dim_size=target_count,
            reduce="sum",
        )
        q_sum = scatter(
            interception * reachable,
            edge_target,
            dim=0,
            dim_size=target_count,
            reduce="sum",
        )
        distance_sum = scatter(
            distance.unsqueeze(-1) * reachable,
            edge_target,
            dim=0,
            dim_size=target_count,
            reduce="sum",
        )

        mean_survival = torch.where(
            has_route,
            survival_sum / safe_count,
            torch.full_like(survival_sum, self.survival_floor),
        ).clamp_min(self.survival_floor)
        mean_q = torch.where(
            has_route,
            q_sum / safe_count,
            torch.ones_like(q_sum),
        )
        mean_distance = torch.where(
            has_route,
            distance_sum / safe_count,
            torch.ones_like(distance_sum),
        )

        demand = target_demand_norm.clamp(0.0, 1.0)
        demand_total = demand.sum(dim=-1)
        demand_denominator = demand_total.clamp_min(1e-8)

        raw = demand_total / float(k)
        survival_adjusted = (
            (demand / mean_survival.unsqueeze(1)).sum(dim=-1)
            * (self.survival_floor / float(k))
        )
        reach_difficulty = (
            demand * (1.0 / (1.0 + reach_count)).unsqueeze(1)
        ).sum(dim=-1) / demand_denominator
        expected_q = (
            demand * mean_q.unsqueeze(1)
        ).sum(dim=-1) / demand_denominator
        expected_distance = (
            demand * mean_distance.unsqueeze(1)
        ).sum(dim=-1) / demand_denominator

        features = torch.stack(
            (
                raw,
                survival_adjusted,
                reach_difficulty,
                expected_q,
                expected_distance,
            ),
            dim=-1,
        )
        features = torch.nan_to_num(
            features, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        if features.shape != (
            target_count,
            self.num_schemes,
            self.scheme_efficiency_dim,
        ):
            raise RuntimeError(
                "scheme efficiency feature shape mismatch: "
                f"{tuple(features.shape)}"
            )
        if hasattr(torch, "_assert_async"):
            torch._assert_async(
                torch.isfinite(features).all(),
                "scheme efficiency features contain NaN/Inf",
            )
        elif not bool(torch.isfinite(features).all()):
            raise FloatingPointError(
                "scheme efficiency features contain NaN/Inf"
            )
        return features

    def _decode_schemes(
        self,
        node_embeddings: Tensor,
        target_mask: Tensor,
        target_demand_norm: Tensor,
        target_damage_norm: Tensor,
        target_threshold_norm: Tensor,
        mode: SchemeMode,
        tau: float,
        gumbel_hard: bool,
        efficiency_features: Tensor | None = None,
        resource_context=None,
        resource_feasible_scheme: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if tau <= 0:
            raise ValueError("scheme temperature tau must be positive")

        target_embeddings = node_embeddings[target_mask]
        t = target_embeddings.size(0)
        l = self.num_schemes

        damage_margin = (
            target_damage_norm - target_threshold_norm.unsqueeze(-1)
        ).unsqueeze(-1)

        descriptors = torch.cat(
            [
                target_demand_norm,
                target_damage_norm.unsqueeze(-1),
                damage_margin,
            ],
            dim=-1,
        )
        scheme_embeddings = self.scheme_encoder(
            descriptors.reshape(t * l, -1)
        ).reshape(t, l, self.scheme_dim)
        if self.use_scheme_efficiency_features:
            if efficiency_features is None:
                raise ValueError("scheme efficiency features are required")
            scheme_embeddings = scheme_embeddings + self.scheme_efficiency_encoder(
                efficiency_features
            )

        expanded_target = target_embeddings.unsqueeze(1).expand(-1, l, -1)
        scheme_logits = self.scheme_scorer(
            torch.cat([expanded_target, scheme_embeddings, resource_context.expand(t,l,-1) if resource_context is not None else expanded_target[..., :0]], dim=-1)
        ).squeeze(-1)

        # G_lj >= G_th_j remains a separate hard physical rule.
        damage_feasible_scheme = (
            target_damage_norm + 1e-6 >= target_threshold_norm.unsqueeze(-1)
        )
        if not bool(torch.all(damage_feasible_scheme.any(dim=-1))):
            raise ValueError("at least one target has no scheme satisfying G_th")

        feasible_scheme = damage_feasible_scheme
        if self.resource_feasible_scheme_mask:
            if resource_feasible_scheme is None:
                raise ValueError("resource-feasible scheme mask is required")
            if resource_feasible_scheme.shape != damage_feasible_scheme.shape:
                raise ValueError("resource-feasible scheme mask shape mismatch")
            joint_feasible = damage_feasible_scheme & resource_feasible_scheme
            # Explicit inference fallback: if every joint candidate is masked,
            # retain the original damage-feasible set for that target.
            fallback_target = ~joint_feasible.any(dim=-1)
            feasible_scheme = torch.where(
                fallback_target.unsqueeze(-1),
                damage_feasible_scheme,
                joint_feasible,
            )

        masked_logits = scheme_logits.masked_fill(~feasible_scheme, -1.0e4)

        if mode == "soft":
            scheme_probs = F.softmax(masked_logits / tau, dim=-1)
        elif mode == "gumbel":
            scheme_probs = F.gumbel_softmax(
                masked_logits,
                tau=tau,
                hard=gumbel_hard,
                dim=-1,
            )
        elif mode == "hard":
            scheme_index = masked_logits.argmax(dim=-1)
            scheme_probs = F.one_hot(
                scheme_index,
                num_classes=self.num_schemes,
            ).to(masked_logits.dtype)
        else:
            raise ValueError(f"unsupported scheme mode: {mode}")

        # Invalid schemes remain exactly zero even after numerical operations.
        scheme_probs = scheme_probs * feasible_scheme.to(scheme_probs.dtype)
        scheme_probs = scheme_probs / scheme_probs.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        return scheme_logits, scheme_probs, scheme_embeddings

    # ------------------------------------------------------------------
    # Stage 2: scheme-conditioned assignment decoder
    # ------------------------------------------------------------------
    def _decode_assignments(
        self,
        data,
        node_embeddings: Tensor,
        target_mask: Tensor,
        inventory_norm: Tensor,
        target_demand_norm: Tensor,
        scheme_probs: Tensor,
        scheme_embeddings: Tensor,
        apply_inventory_projection: bool,
        collect_redistribution_diagnostics: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        k = self.num_weapons
        src, dst = data.edge_index
        edge_attr = data.edge_attr

        target_node_index = torch.nonzero(target_mask, as_tuple=False).view(-1)
        target_lookup = torch.full(
            (data.x.size(0),),
            -1,
            dtype=torch.long,
            device=data.x.device,
        )
        target_lookup[target_node_index] = torch.arange(
            target_node_index.numel(),
            device=data.x.device,
        )
        edge_target_index = target_lookup[dst]
        if bool(torch.any(edge_target_index < 0)):
            raise ValueError("assignment edge points to a non-target node")

        scheme_context = torch.sum(
            scheme_probs.unsqueeze(-1) * scheme_embeddings,
            dim=1,
        )
        expected_demand_norm = torch.einsum(
            "tl,tlk->tk",
            scheme_probs,
            target_demand_norm,
        )
        target_demand = expected_demand_norm * self.demand_max

        pos_src = data.x[src, 0:2]
        pos_dst = data.x[dst, 0:2]
        delta_pos = pos_dst - pos_src
        distance = torch.linalg.vector_norm(
            delta_pos,
            ord=2,
            dim=-1,
            keepdim=True,
        )

        edge_repr = torch.cat(
            [
                node_embeddings[src],
                node_embeddings[dst],
                edge_attr,
                delta_pos,
                distance,
                scheme_context[edge_target_index],
                expected_demand_norm[edge_target_index],
            ],
            dim=-1,
        )
        edge_q = edge_attr[:, 1 : 1 + k]
        edge_x = edge_attr[:, 1 + k : 1 + 2 * k]

        assignment_net_logits = self.assignment_decoder(edge_repr)
        assignment_tactical_bias = None
        if self.tactical_assignment_bias == "distance":
            alpha_d = F.softplus(self.tactical_distance_raw_alpha)
            assignment_tactical_bias = (-alpha_d * edge_attr[:, 0:1]).expand_as(
                assignment_net_logits
            )
        elif self.tactical_assignment_bias == "interception":
            alpha_q = F.softplus(self.tactical_interception_raw_alpha)
            assignment_tactical_bias = -alpha_q * edge_q
        elif self.tactical_assignment_bias == "both":
            alpha_d = F.softplus(self.tactical_distance_raw_alpha)
            alpha_q = F.softplus(self.tactical_interception_raw_alpha)
            assignment_tactical_bias = (
                -alpha_d * edge_attr[:, 0:1] - alpha_q * edge_q
            )
        assignment_logits = (
            assignment_net_logits
            if assignment_tactical_bias is None
            else assignment_net_logits + assignment_tactical_bias
        )

        if self.model_variant == "context_v4":
            distance_cost = distance.squeeze(-1)
            conflict_cost = torch.zeros_like(distance_cost)
            assignment_logits = self.cost_aware_bias(
                assignment_logits,
                distance_cost.unsqueeze(-1),
                edge_q if 'edge_q' in locals() else torch.zeros_like(edge_attr[:,1:1+k]),
                conflict_cost.unsqueeze(-1),
            )

        # X remains the hard physical reachability mask. Phase-6B-1 optionally
        # adds the scenario-dependent I_B mask before base-wise normalization;
        # it never mutates X itself.
        assignment_mask = edge_x
        if self.zero_inventory_assignment_mask:
            inventory_available = (inventory_norm[src] > 0.0).to(edge_x.dtype)
            assignment_mask = edge_x * inventory_available
        masked_logits = assignment_logits.masked_fill(
            assignment_mask <= 0.5,
            -1.0e4,
        )

        # For every (target, weapon), distribute its required demand over bases.
        temperature_scaled_logits = (
            masked_logits
            if self.assignment_temperature == 1.0
            else masked_logits / self.assignment_temperature
        )
        assignment_probs = softmax(temperature_scaled_logits, dst, dim=0)
        assignment_probs = assignment_probs * assignment_mask

        # Defensive renormalization: if malformed data has no reachable route for a
        # weapon-target pair, its probabilities stay zero rather than becoming
        # uniform. This also covers the Phase-6B-1 all-zero-inventory case.
        prob_sum_by_target = scatter(
            assignment_probs,
            dst,
            dim=0,
            dim_size=data.x.size(0),
            reduce="sum",
        )
        assignment_probs = torch.where(
            prob_sum_by_target[dst] > 1e-8,
            assignment_probs / prob_sum_by_target[dst].clamp_min(1e-8),
            torch.zeros_like(assignment_probs),
        )

        survival = (1.0 - edge_q).clamp_min(self.survival_floor)
        demand_on_edge = target_demand[edge_target_index]

        if self.positive_inventory_contention_lambda > 0.0:
            # Phase-6B-2R: exactly one correction from positive-stock overload.
            allocation_first_pass = (
                assignment_probs * demand_on_edge / survival * assignment_mask
            )
            first_pass_load = scatter(
                allocation_first_pass,
                src,
                dim=0,
                dim_size=data.x.size(0),
                reduce="sum",
            )
            inventory = inventory_norm * self.inventory_max
            positive_inventory = inventory > 0.0
            rho = torch.where(
                positive_inventory,
                first_pass_load / inventory.clamp_min(1e-8),
                torch.zeros_like(first_pass_load),
            )
            contention = torch.log1p(F.relu(rho - 1.0))
            corrected_logits = assignment_logits - (
                self.positive_inventory_contention_lambda * contention[src]
            )
            corrected_masked_logits = corrected_logits.masked_fill(
                assignment_mask <= 0.5,
                -1.0e4,
            )
            corrected_scaled_logits = (
                corrected_masked_logits
                if self.assignment_temperature == 1.0
                else corrected_masked_logits / self.assignment_temperature
            )
            assignment_probs = softmax(corrected_scaled_logits, dst, dim=0)
            assignment_probs = assignment_probs * assignment_mask
            corrected_sum = scatter(
                assignment_probs,
                dst,
                dim=0,
                dim_size=data.x.size(0),
                reduce="sum",
            )
            assignment_probs = torch.where(
                corrected_sum[dst] > 1e-8,
                assignment_probs / corrected_sum[dst].clamp_min(1e-8),
                torch.zeros_like(assignment_probs),
            )

        # Before inventory projection:
        #   sum_i A_kij * (1-Q_kij) = expected scheme demand U_kj
        # whenever at least one X_kij route exists.
        allocation_pre_projection = (
            assignment_probs
            * demand_on_edge
            / survival
            * edge_x
        )

        if apply_inventory_projection:
            allocation, inventory_scale = self._project_inventory(
                allocation_pre_projection,
                src,
                inventory_norm,
                data.x.size(0),
            )
        else:
            allocation = allocation_pre_projection
            inventory_scale = torch.ones(
                (data.x.size(0), k),
                dtype=allocation.dtype,
                device=allocation.device,
            )

        redistribution_diagnostics = None
        if self.redistribution_mode != "none":
            allocation, inventory_scale, redistribution_diagnostics = (
                self._redistribute_demand(
                    allocation=allocation,
                    assignment_probs=assignment_probs,
                    target_demand=target_demand,
                    survival=survival,
                    edge_x=edge_x,
                    src=src,
                    edge_target_index=edge_target_index,
                    inventory_norm=inventory_norm,
                    num_nodes=data.x.size(0),
                    collect_diagnostics=collect_redistribution_diagnostics,
                )
            )

        route_load = allocation.sum(dim=-1)

        return (
            assignment_logits,
            assignment_net_logits,
            assignment_tactical_bias,
            assignment_probs,
            allocation_pre_projection,
            allocation,
            route_load,
            inventory_scale,
            edge_target_index,
            redistribution_diagnostics,
        )

    def _project_inventory(
        self,
        allocation: Tensor,
        src: Tensor,
        inventory_norm: Tensor,
        num_nodes: int,
    ) -> tuple[Tensor, Tensor]:
        """Differentiable proportional projection onto base-weapon inventory caps."""
        inventory = inventory_norm * self.inventory_max

        used = scatter(
            allocation,
            src,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )

        scale = torch.where(
            used > inventory + 1e-8,
            inventory / used.clamp_min(1e-8),
            torch.ones_like(used),
        )
        scale = scale.clamp(min=0.0, max=1.0)

        return allocation * scale[src], scale

    def _redistribute_demand(
        self,
        allocation: Tensor,
        assignment_probs: Tensor,
        target_demand: Tensor,
        survival: Tensor,
        edge_x: Tensor,
        src: Tensor,
        edge_target_index: Tensor,
        inventory_norm: Tensor,
        num_nodes: int,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Iteratively reuse residual inventory to reduce target demand shortage."""
        inventory = inventory_norm * self.inventory_max
        target_count = target_demand.size(0)
        eps = 1e-8
        if not collect_diagnostics:
            current = allocation
            inventory_scale = torch.ones_like(inventory)
            for _ in range(self.redistribution_iterations):
                delivered = scatter(
                    current * survival,
                    edge_target_index,
                    dim=0,
                    dim_size=target_count,
                    reduce="sum",
                )
                shortage = (target_demand - delivered).clamp_min(0.0)
                used = scatter(
                    current, src, dim=0, dim_size=num_nodes, reduce="sum"
                )
                remaining = (inventory - used).clamp_min(0.0)
                scores = edge_x * remaining[src] * survival
                if self.redistribution_mode == "network":
                    scores = scores * (assignment_probs + eps).pow(
                        self.redistribution_beta
                    )
                score_total = scatter(
                    scores,
                    edge_target_index,
                    dim=0,
                    dim_size=target_count,
                    reduce="sum",
                )
                weights = torch.where(
                    score_total[edge_target_index] > eps,
                    scores / score_total[edge_target_index].clamp_min(eps),
                    torch.zeros_like(scores),
                )
                delta_raw = (
                    weights
                    * shortage[edge_target_index]
                    / survival.clamp_min(self.survival_floor)
                    * edge_x
                )
                current, inventory_scale = self._project_inventory(
                    current + self.redistribution_eta * delta_raw,
                    src,
                    inventory_norm,
                    num_nodes,
                )
            return current, inventory_scale, None

        initial = allocation
        initial_active = initial.sum(dim=-1) > eps

        initial_used = scatter(
            initial, src, dim=0, dim_size=num_nodes, reduce="sum"
        )
        initial_remaining = (inventory - initial_used).clamp_min(0.0)
        max_probability = scatter(
            assignment_probs,
            edge_target_index,
            dim=0,
            dim_size=target_count,
            reduce="max",
        )
        dominant = (
            assignment_probs >= max_probability[edge_target_index] - eps
        ) & (edge_x > 0.5)

        normalized_shortage = []
        shortage_pairs = []
        shortage_targets = []
        shots_per_iteration = []
        arrivals_per_iteration = []
        cumulative_positive = torch.zeros_like(allocation)

        def shortage_state(current: Tensor) -> tuple[Tensor, Tensor]:
            delivered = scatter(
                current * survival,
                edge_target_index,
                dim=0,
                dim_size=target_count,
                reduce="sum",
            )
            shortage = (target_demand - delivered).clamp_min(0.0)
            return shortage, delivered

        initial_shortage, _ = shortage_state(initial)
        initial_effective_capacity = scatter(
            edge_x * initial_remaining[src] * survival,
            edge_target_index,
            dim=0,
            dim_size=target_count,
            reduce="sum",
        )
        initial_recoverable = torch.minimum(
            initial_shortage, initial_effective_capacity
        ).sum()

        current = allocation
        for _ in range(self.redistribution_iterations):
            shortage, _ = shortage_state(current)
            normalized_shortage.append(
                shortage.sum() / target_demand.sum().clamp_min(eps)
            )
            shortage_pairs.append((shortage > eps).sum().to(current.dtype))
            shortage_targets.append(
                (shortage > eps).any(dim=-1).sum().to(current.dtype)
            )

            used = scatter(
                current, src, dim=0, dim_size=num_nodes, reduce="sum"
            )
            remaining = (inventory - used).clamp_min(0.0)
            effective_capacity = edge_x * remaining[src] * survival
            if self.redistribution_mode == "capacity":
                scores = effective_capacity
            else:
                scores = effective_capacity * (
                    assignment_probs + eps
                ).pow(self.redistribution_beta)
            score_total = scatter(
                scores,
                edge_target_index,
                dim=0,
                dim_size=target_count,
                reduce="sum",
            )
            weights = torch.where(
                score_total[edge_target_index] > eps,
                scores / score_total[edge_target_index].clamp_min(eps),
                torch.zeros_like(scores),
            )
            delta_raw = (
                weights
                * shortage[edge_target_index]
                / survival.clamp_min(self.survival_floor)
                * edge_x
            )
            candidate = current + self.redistribution_eta * delta_raw
            updated, inventory_scale = self._project_inventory(
                candidate, src, inventory_norm, num_nodes
            )
            positive = (updated - current).clamp_min(0.0)
            cumulative_positive = cumulative_positive + positive
            shots_per_iteration.append(positive.sum())
            arrivals_per_iteration.append((positive * survival).sum())
            current = updated

        final_shortage, _ = shortage_state(current)
        normalized_shortage.append(
            final_shortage.sum() / target_demand.sum().clamp_min(eps)
        )
        shortage_pairs.append((final_shortage > eps).sum().to(current.dtype))
        shortage_targets.append(
            (final_shortage > eps).any(dim=-1).sum().to(current.dtype)
        )
        final_used = scatter(
            current, src, dim=0, dim_size=num_nodes, reduce="sum"
        )
        final_remaining = (inventory - final_used).clamp_min(0.0)
        positive_total = cumulative_positive.sum().clamp_min(eps)
        route_positive = cumulative_positive.sum(dim=-1)
        diagnostics = {
            "recoverable_ratio_initial": (
                initial_recoverable / initial_shortage.sum().clamp_min(eps)
            ).detach(),
            "normalized_shortage": torch.stack(normalized_shortage).detach(),
            "shortage_pairs": torch.stack(shortage_pairs).detach(),
            "shortage_targets": torch.stack(shortage_targets).detach(),
            "redistributed_shots_per_iteration": torch.stack(
                shots_per_iteration
            ).detach(),
            "effective_arrivals_per_iteration": torch.stack(
                arrivals_per_iteration
            ).detach(),
            "redistributed_shots": cumulative_positive.sum().detach(),
            "effective_redistributed_arrivals": (
                cumulative_positive * survival
            ).sum().detach(),
            "fraction_to_previously_active_routes": (
                cumulative_positive[initial_active].sum() / positive_total
            ).detach(),
            "fraction_to_previously_inactive_routes": (
                cumulative_positive[~initial_active].sum() / positive_total
            ).detach(),
            "newly_activated_routes": (
                (~initial_active) & (current.sum(dim=-1) > eps)
            ).sum().to(current.dtype).detach(),
            "remaining_unused_inventory": final_remaining.sum().detach(),
            "fraction_to_underutilized_base_weapons": (
                cumulative_positive[
                    initial_remaining[src] > eps
                ].sum() / positive_total
            ).detach(),
            "fraction_to_original_dominant_routes": (
                cumulative_positive[dominant].sum() / positive_total
            ).detach(),
            "route_positive_allocation": route_positive.detach(),
        }
        return current, inventory_scale, diagnostics

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(
        self,
        data,
        scheme_mode: SchemeMode | None = None,
        scheme_tau: float = 1.0,
        gumbel_hard: bool = False,
        apply_inventory_projection: bool | None = None,
        collect_redistribution_diagnostics: bool = False,
    ) -> DeepWTAOutput:
        (
            base_mask,
            target_mask,
            inventory_norm,
            target_demand_norm,
            target_damage_norm,
            target_threshold_norm,
        ) = self._split_node_features(data.x)

        self._validate_graph(data, base_mask, target_mask)

        if scheme_mode is None:
            # Stable continuous relaxation during training; deterministic scheme
            # selection during validation/inference.
            scheme_mode = "soft" if self.training else "hard"

        if apply_inventory_projection is None:
            apply_inventory_projection = self.use_inventory_projection

        node_embeddings = self.encode(data)
        resource_context = self._resource_context(data)
        scheme_efficiency_features = None
        if self.use_scheme_efficiency_features:
            scheme_efficiency_features = self._scheme_efficiency_features(
                data,
                target_mask,
                target_demand_norm,
            )

        resource_feasible_scheme = None
        if self.resource_feasible_scheme_mask:
            resource_feasible_scheme = self._resource_feasible_schemes(
                data,
                target_mask,
                inventory_norm,
                target_demand_norm,
            )

        scheme_logits, scheme_probs, scheme_embeddings = self._decode_schemes(
            node_embeddings=node_embeddings,
            target_mask=target_mask,
            target_demand_norm=target_demand_norm,
            target_damage_norm=target_damage_norm,
            target_threshold_norm=target_threshold_norm,
            mode=scheme_mode,
            tau=scheme_tau,
            gumbel_hard=gumbel_hard,
            efficiency_features=scheme_efficiency_features,
            resource_context=resource_context,
            resource_feasible_scheme=resource_feasible_scheme,
        )

        (
            assignment_logits,
            assignment_net_logits,
            assignment_tactical_bias,
            assignment_probs,
            allocation_pre_projection,
            allocation,
            route_load,
            inventory_scale,
            edge_target_index,
            redistribution_diagnostics,
        ) = self._decode_assignments(
            data=data,
            node_embeddings=node_embeddings,
            target_mask=target_mask,
            inventory_norm=inventory_norm,
            target_demand_norm=target_demand_norm,
            scheme_probs=scheme_probs,
            scheme_embeddings=scheme_embeddings,
            apply_inventory_projection=apply_inventory_projection,
            collect_redistribution_diagnostics=collect_redistribution_diagnostics,
        )

        target_node_index = torch.nonzero(target_mask, as_tuple=False).view(-1)

        return DeepWTAOutput(
            allocation=allocation,
            allocation_pre_projection=allocation_pre_projection,
            assignment_probs=assignment_probs,
            assignment_logits=assignment_logits,
            scheme_probs=scheme_probs,
            scheme_logits=scheme_logits,
            target_demand=(
                torch.einsum(
                    "tl,tlk->tk",
                    scheme_probs,
                    target_demand_norm,
                )
                * self.demand_max
            ),
            route_load=route_load,
            target_node_index=target_node_index,
            edge_target_index=edge_target_index,
            inventory_scale=inventory_scale,
            scheme_efficiency_features=scheme_efficiency_features,
            assignment_net_logits=assignment_net_logits,
            assignment_tactical_bias=assignment_tactical_bias,
            redistribution_diagnostics=redistribution_diagnostics,
        )
