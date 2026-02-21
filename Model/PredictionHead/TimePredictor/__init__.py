import torch
import torch.nn as nn

class TimePredictor(nn.Module):
    def __init__(self, embedding_dim:int, context_dim:int):
        super().__init__()
        self.layer_01 = nn.Sequential(
            nn.Linear(embedding_dim + context_dim, 16),
            nn.ReLU()
        )
        self.layer_02 = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.layer_03 = nn.Sequential(
            nn.Linear(8, 4),
            nn.Sigmoid() # Output between 0 and 1 for one hour risk prediction
        )

    def forward(self, embedding: torch.Tensor,context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([embedding, context], dim=-1)  # Concatenate along feature dimension
        x = self.layer_01(x)
        x = self.layer_02(x)
        x = self.layer_03(x)
        return x