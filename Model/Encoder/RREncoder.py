import torch.nn as nn
import torch
from Configs import LATENT_SIZE

class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_SIZE):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(32, latent_dim)

    def forward(self, rr_window)->torch.Tensor:
        """
        rr_window: [B, 50]
        output:    [B, LATENT_SIZE]
        """
        x = rr_window.unsqueeze(1)
        h = self.encoder(x).squeeze(-1)
        z = self.proj(h)
        return z
