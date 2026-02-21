import torch.nn as nn
import torch

class HRVPredictor(nn.Module):
    def __init__(self, context_dim:int, num_targets:int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_targets)
        )

    def forward(self, c_t:torch.Tensor)->torch.Tensor:
        """
        c_t: [B, context_dim]
        output: [B, num_targets]
        """
        return self.net(c_t)