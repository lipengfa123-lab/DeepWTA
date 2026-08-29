import torch
import torch.nn as nn

class ResourceContextEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(3,64), nn.ReLU(),
            nn.Linear(64,out_dim), nn.ReLU()
        )
    def forward(self, scarcity, accessibility, pressure):
        x=torch.stack([scarcity,accessibility,pressure],dim=-1)
        return self.net(x).mean(dim=0)
