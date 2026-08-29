"""Graph-attention encoder components used by DeepWTA."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv


class ResidualGraphAttentionBlock(nn.Module):
    """GATv2 block with residual projection, LayerNorm and ELU activation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        heads: int,
        edge_dim: int,
        dropout: float,
    ):
        super().__init__()
        out_dim = hidden_dim * heads
        self.conv = GATv2Conv(
            in_dim,
            hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        hidden = self.conv(x, edge_index, edge_attr=edge_attr)
        hidden = self.norm(hidden + self.skip(x))
        return self.dropout(F.elu(hidden))


# Checkpoint-era source imports used this shorter name.
ResidualGATBlock = ResidualGraphAttentionBlock

