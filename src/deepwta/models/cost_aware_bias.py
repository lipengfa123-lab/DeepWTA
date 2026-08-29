import torch
import torch.nn as nn
import torch.nn.functional as F

class CostAwareBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_distance=nn.Parameter(torch.tensor(0.5).log())
        self.log_interception=nn.Parameter(torch.tensor(0.5).log())
        self.log_conflict=nn.Parameter(torch.tensor(0.3).log())
    def values(self):
        return {
            "alpha_distance":F.softplus(self.log_distance),
            "alpha_interception":F.softplus(self.log_interception),
            "alpha_conflict":F.softplus(self.log_conflict),
        }
    def forward(self, logits, distance, interception, conflict):
        a=self.values()
        return logits-a["alpha_distance"]*distance-a["alpha_interception"]*interception-a["alpha_conflict"]*conflict
