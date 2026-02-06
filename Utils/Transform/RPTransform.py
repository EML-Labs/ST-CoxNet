import torch
import torch.nn as nn

class RPTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, dim, T)
        return: (B, C, T, T)  -- distance-based recurrence plot
        """
        B, C, dim, T = x.shape

        # (B, C, T, dim)
        x_t = x.permute(0, 1, 3, 2)

        # Expand for pairwise subtraction
        xi = x_t.unsqueeze(3)  # (B, C, T, 1, dim)
        xj = x_t.unsqueeze(2)  # (B, C, 1, T, dim)

        # Euclidean distance
        rp = torch.norm(xi - xj, dim=-1)  # (B, C, T, T)

        return rp      