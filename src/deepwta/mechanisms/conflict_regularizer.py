from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from deepwta.data.trajectory_data import (
    broad_phase_filter,
    build_dual_edge_features,
    build_trajectory_features,
    compute_exact_cross_label,
)
from deepwta.models.trajectory_conflict_gnn import TrajectoryConflictGNN


@dataclass
class ConflictRegularizerResult:
    loss: Tensor
    mean_expected_count: Tensor
    mean_propensity: Tensor
    mean_route_activation: Tensor
    mean_route_coverage: Tensor
    mean_candidate_pairs: Tensor
    graphs_with_pairs: int


def load_frozen_conflict_gnn(
    checkpoint_path: str | Path,
    device: torch.device | str,
    temperature_mode: str = "calibrated",
) -> tuple[TrajectoryConflictGNN, float]:
    """Load and freeze the v5 trajectory-conflict surrogate.

    `temperature_mode="calibrated"` intentionally uses the scalar fitted on
    the surrogate validation set. It does not rely on the temperature buffer
    stored inside the checkpoint, because the final checkpoint may store a
    separate deployment temperature.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"conflict checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model_cfg = checkpoint["model_config"]
    model = TrajectoryConflictGNN(
        node_dim=int(model_cfg["node_dim"]),
        edge_dim=int(model_cfg["edge_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        heads=int(model_cfg["heads"]),
        edge_mlp_dim=int(model_cfg["edge_mlp_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if temperature_mode == "calibrated":
        temperature = checkpoint.get("calibrated_temperature")
        if temperature is None:
            raise ValueError(
                "checkpoint does not contain calibrated_temperature"
            )
    elif temperature_mode == "raw":
        temperature = 1.0
    elif temperature_mode == "deployment":
        temperature = checkpoint.get("deployment_temperature")
        if temperature is None:
            raise ValueError(
                "checkpoint does not contain deployment_temperature"
            )
    else:
        raise ValueError(
            "temperature_mode must be one of: calibrated, raw, deployment"
        )

    return model, float(temperature)


class FrozenConflictRegularizer(nn.Module):
    """Continuous crossing regularizer driven by a frozen Conflict GNN.

    Main-model routes are never selected by thresholding `A_soft`.
    For large WTA graphs, a geometry-only stratified route sample is used before
    deterministic Sweep-and-Prune/AABB candidate pruning. The route sample changes
    with `sampling_step` but is independent of the current allocation values.

    The differentiable path is:
        A_soft -> route_load -> soft route activation -> crossing loss.
    The frozen surrogate only supplies fixed continuous coefficients p_rs.
    """

    def __init__(
        self,
        surrogate: TrajectoryConflictGNN,
        num_weapons: int,
        map_height: int,
        map_width: int,
        temperature: float,
        route_activation_tau: float = 1.0,
        max_routes_per_graph: int = 72,
        route_sample_repeats: int = 1,
        sample_seed: int = 2026,
    ):
        super().__init__()

        if route_activation_tau <= 0:
            raise ValueError("route_activation_tau must be positive")
        if max_routes_per_graph < 2:
            raise ValueError("max_routes_per_graph must be at least 2")
        if route_sample_repeats < 1:
            raise ValueError("route_sample_repeats must be at least 1")
        if map_height <= 0 or map_width <= 0:
            raise ValueError("map dimensions must be positive")
        if temperature <= 0:
            raise ValueError("surrogate temperature must be positive")

        self.surrogate = surrogate
        self.num_weapons = int(num_weapons)
        self.map_height = int(map_height)
        self.map_width = int(map_width)
        self.temperature = float(temperature)
        self.route_activation_tau = float(route_activation_tau)
        self.max_routes_per_graph = int(max_routes_per_graph)
        self.route_sample_repeats = int(route_sample_repeats)
        self.sample_seed = int(sample_seed)

        self.surrogate.eval()
        for parameter in self.surrogate.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        # The regularizer itself has no trainable parameters. Keep the surrogate
        # in eval mode even if a parent criterion is put into train mode.
        super().train(mode)
        self.surrogate.eval()
        return self

    @staticmethod
    def soft_route_activation(route_load: Tensor, tau: float) -> Tensor:
        load_fp32 = route_load.float().clamp_min(0.0)
        return 1.0 - torch.exp(-load_fp32 / tau)

    @staticmethod
    def _stable_scenario_key(
        base_coords: np.ndarray,
        target_coords: np.ndarray,
    ) -> int:
        # Deterministic geometry-derived key; independent of Python hash randomization.
        coords = np.concatenate([base_coords.reshape(-1), target_coords.reshape(-1)])
        if coords.size == 0:
            return 0
        weights = ((np.arange(coords.size, dtype=np.int64) % 97) + 1).astype(np.float64)
        value = float(np.sum(coords.astype(np.float64) * weights))
        return int(abs(round(value * 1000.0))) % 2_000_000_000

    def _stratified_route_sample(
        self,
        routes: np.ndarray,
        eligible_indices: np.ndarray,
        generator: np.random.Generator,
    ) -> np.ndarray:
        """Sample routes without using model outputs, preserving target diversity."""
        if len(eligible_indices) <= self.max_routes_per_graph:
            return eligible_indices.copy()

        route_targets = routes[eligible_indices, 1]
        unique_targets = np.unique(route_targets)
        generator.shuffle(unique_targets)

        chosen: list[int] = []
        chosen_set: set[int] = set()

        # First pass: one route per target whenever the cap permits.
        for target in unique_targets:
            members = eligible_indices[route_targets == target]
            idx = int(generator.choice(members))
            chosen.append(idx)
            chosen_set.add(idx)
            if len(chosen) >= self.max_routes_per_graph:
                break

        if len(chosen) < self.max_routes_per_graph:
            remaining = np.array(
                [idx for idx in eligible_indices if int(idx) not in chosen_set],
                dtype=np.int64,
            )
            if len(remaining) > 0:
                need = min(
                    self.max_routes_per_graph - len(chosen),
                    len(remaining),
                )
                extra = generator.choice(remaining, size=need, replace=False)
                chosen.extend(int(v) for v in extra)

        return np.asarray(chosen, dtype=np.int64)

    def _build_local_geometry(
        self,
        batch,
        graph_index: int,
        edge_positions: Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        node_batch = (
            batch.batch
            if hasattr(batch, "batch") and batch.batch is not None
            else torch.zeros(
                batch.x.size(0),
                dtype=torch.long,
                device=batch.x.device,
            )
        )

        graph_nodes = torch.nonzero(node_batch == graph_index, as_tuple=False).reshape(-1)
        node_type = batch.x[graph_nodes, 2]
        base_nodes = graph_nodes[node_type < 0.5]
        target_nodes = graph_nodes[node_type >= 0.5]

        # x stores [x_norm, y_norm]. trajectory_data expects [row(y), col(x)].
        base_x = batch.x[base_nodes, 0].detach().float().cpu().numpy()
        base_y = batch.x[base_nodes, 1].detach().float().cpu().numpy()
        target_x = batch.x[target_nodes, 0].detach().float().cpu().numpy()
        target_y = batch.x[target_nodes, 1].detach().float().cpu().numpy()

        base_coords = np.column_stack(
            [base_y * self.map_height, base_x * self.map_width]
        ).astype(np.float32)
        target_coords = np.column_stack(
            [target_y * self.map_height, target_x * self.map_width]
        ).astype(np.float32)

        base_lookup = {
            int(global_id): local_id
            for local_id, global_id in enumerate(base_nodes.detach().cpu().tolist())
        }
        target_lookup = {
            int(global_id): local_id
            for local_id, global_id in enumerate(target_nodes.detach().cpu().tolist())
        }

        src = batch.edge_index[0, edge_positions].detach().cpu().tolist()
        dst = batch.edge_index[1, edge_positions].detach().cpu().tolist()
        routes = np.asarray(
            [
                [base_lookup[int(s)], target_lookup[int(d)]]
                for s, d in zip(src, dst)
            ],
            dtype=np.int64,
        )

        return base_coords, target_coords, routes, graph_nodes.detach().cpu().numpy()

    def _score_route_subset(
        self,
        base_coords: np.ndarray,
        target_coords: np.ndarray,
        routes: np.ndarray,
        selected_local_indices: np.ndarray,
        selected_activation: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        selected_routes = routes[selected_local_indices]
        pairs, _ = broad_phase_filter(
            base_coords,
            target_coords,
            selected_routes,
        )

        if len(pairs) == 0:
            zero = selected_activation.sum() * 0.0
            return zero, zero, zero, 0

        node_x = build_trajectory_features(
            base_coords,
            target_coords,
            selected_routes,
            self.map_height,
            self.map_width,
        )
        forward_attr = build_dual_edge_features(
            base_coords,
            target_coords,
            selected_routes,
            pairs,
            self.map_height,
            self.map_width,
        )

        reverse_pairs = pairs[:, ::-1]
        reverse_attr = build_dual_edge_features(
            base_coords,
            target_coords,
            selected_routes,
            reverse_pairs,
            self.map_height,
            self.map_width,
        )

        dual_edge_index = np.concatenate(
            [pairs.T, reverse_pairs.T],
            axis=1,
        ).astype(np.int64)
        dual_edge_attr = np.concatenate(
            [forward_attr, reverse_attr],
            axis=0,
        ).astype(np.float32)

        dual_graph = Data(
            x=torch.from_numpy(node_x).to(device),
            edge_index=torch.from_numpy(dual_edge_index).to(device),
            edge_attr=torch.from_numpy(dual_edge_attr).to(device),
        )

        self.surrogate.eval()
        # Keep the already calibrated surrogate in FP32 even when the main WTA
        # forward runs under AMP. p_rs is a frozen coefficient; there is no need
        # to trade calibration quality for surrogate AMP speed here.
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            enabled=False,
        ):
            _, directed_probs, _, _ = self.surrogate(
                dual_graph,
                temperature=self.temperature,
            )

        pair_count = len(pairs)
        propensity = 0.5 * (
            directed_probs[:pair_count] + directed_probs[pair_count:]
        )
        propensity = propensity.detach()

        pair_i = torch.from_numpy(pairs[:, 0]).to(
            device=device,
            dtype=torch.long,
        )
        pair_j = torch.from_numpy(pairs[:, 1]).to(
            device=device,
            dtype=torch.long,
        )

        contribution = (
            propensity
            * selected_activation[pair_i]
            * selected_activation[pair_j]
        )

        return (
            contribution.mean(),
            contribution.sum(),
            propensity.mean(),
            pair_count,
        )

    def forward(
        self,
        batch,
        route_load: Tensor,
        sampling_step: int = 0,
    ) -> ConflictRegularizerResult:
        if route_load.ndim != 1:
            raise ValueError("route_load must have shape [E]")
        if route_load.numel() != batch.edge_index.size(1):
            raise ValueError(
                "route_load must align with the original WTA edge_index"
            )

        device = route_load.device
        k = self.num_weapons
        edge_x = batch.edge_attr[:, 1 + k : 1 + 2 * k]
        reachable_route = torch.any(edge_x > 0.5, dim=-1)

        node_batch = (
            batch.batch
            if hasattr(batch, "batch") and batch.batch is not None
            else torch.zeros(
                batch.x.size(0),
                dtype=torch.long,
                device=batch.x.device,
            )
        )
        src = batch.edge_index[0]
        edge_batch = node_batch[src]
        num_graphs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else 1

        activation = self.soft_route_activation(
            route_load,
            self.route_activation_tau,
        )

        graph_losses: list[Tensor] = []
        graph_counts: list[Tensor] = []
        graph_propensities: list[Tensor] = []
        graph_activations: list[Tensor] = []
        coverages: list[Tensor] = []
        candidate_counts: list[Tensor] = []
        graphs_with_pairs = 0

        for graph_index in range(num_graphs):
            graph_edge_positions = torch.nonzero(edge_batch == graph_index, as_tuple=False).reshape(-1)
            if graph_edge_positions.numel() < 2:
                continue

            local_reachable = reachable_route[graph_edge_positions]
            eligible_local = torch.nonzero(local_reachable, as_tuple=False).reshape(-1).detach().cpu().numpy()
            if len(eligible_local) < 2:
                continue

            base_coords, target_coords, routes, _ = self._build_local_geometry(
                batch,
                graph_index,
                graph_edge_positions,
            )
            scenario_key = self._stable_scenario_key(
                base_coords,
                target_coords,
            )

            repeat_losses: list[Tensor] = []
            repeat_counts: list[Tensor] = []
            repeat_props: list[Tensor] = []
            repeat_acts: list[Tensor] = []
            repeat_pairs: list[Tensor] = []
            repeat_coverages: list[Tensor] = []

            for repeat in range(self.route_sample_repeats):
                seed = (
                    self.sample_seed
                    + int(sampling_step) * 1_000_003
                    + scenario_key * 101
                    + repeat * 10_007
                ) % (2**32 - 1)
                rng = np.random.default_rng(seed)

                selected_local = self._stratified_route_sample(
                    routes,
                    eligible_local,
                    rng,
                )
                if len(selected_local) < 2:
                    continue

                selected_edge_positions = graph_edge_positions[
                    torch.from_numpy(selected_local).to(
                        graph_edge_positions.device,
                        dtype=torch.long,
                    )
                ]
                selected_activation = activation[selected_edge_positions]

                loss, expected_count, mean_prop, pair_count = self._score_route_subset(
                    base_coords=base_coords,
                    target_coords=target_coords,
                    routes=routes,
                    selected_local_indices=selected_local,
                    selected_activation=selected_activation,
                    device=device,
                )
                if pair_count == 0:
                    continue

                repeat_losses.append(loss)
                repeat_counts.append(expected_count)
                repeat_props.append(mean_prop)
                repeat_acts.append(selected_activation.mean())
                repeat_pairs.append(
                    torch.tensor(
                        float(pair_count),
                        dtype=route_load.dtype,
                        device=device,
                    )
                )
                repeat_coverages.append(
                    torch.tensor(
                        len(selected_local) / max(len(eligible_local), 1),
                        dtype=route_load.dtype,
                        device=device,
                    )
                )

            if not repeat_losses:
                continue

            graphs_with_pairs += 1
            graph_losses.append(torch.stack(repeat_losses).mean())
            graph_counts.append(torch.stack(repeat_counts).mean())
            graph_propensities.append(torch.stack(repeat_props).mean())
            graph_activations.append(torch.stack(repeat_acts).mean())
            candidate_counts.append(torch.stack(repeat_pairs).mean())
            coverages.append(torch.stack(repeat_coverages).mean())

        if not graph_losses:
            zero = route_load.sum() * 0.0
            return ConflictRegularizerResult(
                loss=zero,
                mean_expected_count=zero,
                mean_propensity=zero,
                mean_route_activation=zero,
                mean_route_coverage=zero,
                mean_candidate_pairs=zero,
                graphs_with_pairs=0,
            )

        return ConflictRegularizerResult(
            loss=torch.stack(graph_losses).mean(),
            mean_expected_count=torch.stack(graph_counts).mean(),
            mean_propensity=torch.stack(graph_propensities).mean(),
            mean_route_activation=torch.stack(graph_activations).mean(),
            mean_route_coverage=torch.stack(coverages).mean(),
            mean_candidate_pairs=torch.stack(candidate_counts).mean(),
            graphs_with_pairs=graphs_with_pairs,
        )



class ExactBinaryConflictRegularizer(FrozenConflictRegularizer):
    """Exact-C ablation using C_rs in {0,1} with the SAME sampled route graph.

    This intentionally keeps:
      - geometry-only route sampling,
      - target-stratified route cap,
      - Sweep-and-Prune/AABB broad phase,
      - soft route activation a_r = 1-exp(-s_r/tau),
      - aggregation and normalization

    identical to FrozenConflictRegularizer.

    The only controlled change is:
        learned p_rs  ->  exact binary C_rs.

    Therefore Full-vs-Exact-C directly tests whether the learned continuous
    conflict propensity contributes beyond the exact binary crossing relation.
    """

    def __init__(
        self,
        num_weapons: int,
        map_height: int,
        map_width: int,
        route_activation_tau: float = 1.0,
        max_routes_per_graph: int = 72,
        route_sample_repeats: int = 1,
        sample_seed: int = 2026,
    ):
        super().__init__(
            surrogate=nn.Identity(),
            num_weapons=num_weapons,
            map_height=map_height,
            map_width=map_width,
            temperature=1.0,
            route_activation_tau=route_activation_tau,
            max_routes_per_graph=max_routes_per_graph,
            route_sample_repeats=route_sample_repeats,
            sample_seed=sample_seed,
        )

    def _score_route_subset(
        self,
        base_coords: np.ndarray,
        target_coords: np.ndarray,
        routes: np.ndarray,
        selected_local_indices: np.ndarray,
        selected_activation: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        selected_routes = routes[selected_local_indices]
        pairs, _ = broad_phase_filter(
            base_coords,
            target_coords,
            selected_routes,
        )

        if len(pairs) == 0:
            zero = selected_activation.sum() * 0.0
            return zero, zero, zero, 0

        labels = compute_exact_cross_label(
            base_coords,
            target_coords,
            selected_routes,
            pairs,
        ).astype(np.float32)

        coefficient = torch.from_numpy(labels).to(
            device=device,
            dtype=torch.float32,
        )
        pair_i = torch.from_numpy(pairs[:, 0]).to(
            device=device,
            dtype=torch.long,
        )
        pair_j = torch.from_numpy(pairs[:, 1]).to(
            device=device,
            dtype=torch.long,
        )

        contribution = (
            coefficient
            * selected_activation[pair_i]
            * selected_activation[pair_j]
        )

        return (
            contribution.mean(),
            contribution.sum(),
            coefficient.mean(),
            int(len(pairs)),
        )
