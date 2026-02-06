import torch
import torch.nn as nn

class RPMinMaxTransform(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, rp: torch.Tensor) -> torch.Tensor:
        """
        rp: (B, C, T, T)
        """
        min_val = rp.amin(dim=(2, 3), keepdim=True)
        max_val = rp.amax(dim=(2, 3), keepdim=True)

        return (rp - min_val) / (max_val - min_val + self.eps)
