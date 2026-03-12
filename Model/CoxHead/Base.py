from torch import nn
import torch
class CoxHead(nn.Module):
    def __init__(self, context_dim:int,latent_dim:int):
        super().__init__()
        input_dim = context_dim + latent_dim
        self.context_norm = nn.LayerNorm(context_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    def forward(self, c_last, z_last):
        context = self.context_norm(c_last)
        latent = self.latent_norm(z_last)
        combined = torch.cat([context, latent], dim=1)
        return self.net(combined)
