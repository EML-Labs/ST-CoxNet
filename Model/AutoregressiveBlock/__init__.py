import torch
import torch.nn as nn
from Configs import LATENT_SIZE, CONTEXT_SIZE

class ARBlock(nn.Module):
    def __init__(self, latent_dim=LATENT_SIZE, context_dim=CONTEXT_SIZE):
        super().__init__()

        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=context_dim,
            batch_first=True
        )

    def forward(self, z_seq)->torch.Tensor:
        """
        z_seq: [B, T, LATENT_SIZE]
        output: [B, T, CONTEXT_SIZE]
        """
        c_seq, _ = self.gru(z_seq)
        return c_seq   # [B, T, CONTEXT_SIZE]
