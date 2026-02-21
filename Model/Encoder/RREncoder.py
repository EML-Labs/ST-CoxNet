import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, latent_dim:int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2,stride=2), # [B, 1, 50] -> [B, 8, 25]
            nn.GroupNorm(2,8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, padding=2,stride=2), # [B, 8, 25] -> [B, 16, 13]
            nn.GroupNorm(4,16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2,stride=2), # [B, 16, 13] -> [B, 32, 7]
            nn.GroupNorm(8,32),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7, latent_dim),
            nn.LayerNorm(latent_dim)
            )

    def forward(self, rr_window)->torch.Tensor:
        """
        rr_window: [B, 50]
        output:    [B, LATENT_SIZE]
        """
        x = rr_window.unsqueeze(1)
        h = self.encoder(x)  # [B, 32, 7]
        z = self.proj(h)
        return z
