import torch.nn as nn
import torch

class SymmetricLoss(nn.Module):
    def __init__(self):
        super(SymmetricLoss, self).__init__()

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        # prediction: [B, C, H, W]
        if prediction.dim() != 4:
            raise ValueError("Input tensor must be 4-dimensional [B, C, H, W]")
        prediction = prediction.squeeze(1)  # [B, H, W]

        # Difference between upper and lower triangle
        diff = prediction - prediction.transpose(-1, -2)
        # Only take upper triangle without diagonal
        triu_idx = torch.triu_indices(diff.shape[-2], diff.shape[-1], offset=1)
        diff_upper = diff[:, triu_idx[0], triu_idx[1]]
        return diff_upper.abs().mean()