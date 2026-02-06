import torch
import torch.nn as nn

class RPZNormTransform(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, rp: torch.Tensor) -> torch.Tensor:
        """
        rp: (B, C, T, T)
        """
        mean = rp.mean(dim=(2, 3), keepdim=True)
        std  = rp.std(dim=(2, 3), keepdim=True)

        return (rp - mean) / (std + self.eps)