from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATv2Conv, global_add_pool


class TrajectoryConflictGNN(nn.Module):
    """Predict continuous conflict propensity for pruned route pairs."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        edge_mlp_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)

        width = hidden_dim * heads

        self.conv1 = GATv2Conv(node_dim, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.norm1 = BatchNorm(width)
        self.skip1 = nn.Linear(node_dim, width)

        self.conv2 = GATv2Conv(width, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.norm2 = BatchNorm(width)
        self.skip2 = nn.Linear(width, width)

        self.conv3 = GATv2Conv(width, hidden_dim, heads=1, edge_dim=edge_dim)
        self.norm3 = BatchNorm(hidden_dim)
        self.skip3 = nn.Linear(width, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, edge_mlp_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_mlp_dim, edge_mlp_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_mlp_dim // 2, 2),
        )

        # Saved in the state_dict so DeepWTA can reuse the deployment temperature.
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def set_temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError("temperature must be positive")
        self.temperature.fill_(float(value))

    def encode(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = F.elu(self.norm1(h) + self.skip1(x))
        h = self.dropout(h)

        h2 = self.conv2(h, edge_index, edge_attr=edge_attr)
        h2 = F.elu(self.norm2(h2) + self.skip2(h))
        h2 = self.dropout(h2)

        h3 = self.conv3(h2, edge_index, edge_attr=edge_attr)
        return F.elu(self.norm3(h3) + self.skip3(h2))

    def forward(self, data, temperature: float | None = None):
        node_embeddings = self.encode(data)
        src, dst = data.edge_index

        edge_repr = torch.cat(
            [node_embeddings[src], node_embeddings[dst], data.edge_attr],
            dim=-1,
        )
        edge_logits = self.edge_predictor(edge_repr)

        temp = self.temperature if temperature is None else torch.as_tensor(
            temperature,
            dtype=edge_logits.dtype,
            device=edge_logits.device,
        )
        temp = torch.clamp(temp, min=1e-4)
        edge_probs = F.softmax(edge_logits / temp, dim=-1)[:, 1]

        if hasattr(data, "batch") and data.batch is not None:
            node_batch = data.batch
        else:
            node_batch = torch.zeros(
                data.x.size(0),
                dtype=torch.long,
                device=data.x.device,
            )

        edge_batch = node_batch[src]
        prob_sum = global_add_pool(edge_probs.unsqueeze(-1), edge_batch).squeeze(-1)
        edge_count = global_add_pool(
            torch.ones_like(edge_probs).unsqueeze(-1),
            edge_batch,
        ).squeeze(-1)

        # Every undirected candidate pair is stored as two directed GAT edges.
        graph_cross_count = prob_sum / 2.0
        graph_pair_count = torch.clamp(edge_count / 2.0, min=1.0)

        return edge_logits, edge_probs, graph_cross_count, graph_pair_count


# Temporary compatibility alias for older imports. New code should use the explicit name.
EdgeGATNet = TrajectoryConflictGNN
