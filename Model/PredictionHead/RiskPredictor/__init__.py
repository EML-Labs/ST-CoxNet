import torch.nn as nn
import torch

class RiskPredictor(nn.Module):
    def __init__(self, emb_dim:int=16, output_dim:int=1):
        super().__init__()
        self.fc = nn.Linear(emb_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)