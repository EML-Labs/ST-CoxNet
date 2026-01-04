
import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        prediction :  [B, C, H, W] Predicted output from the model
        target     :  [B, C, H, W] Ground truth images
        Returns the SSIM loss between prediction and target.
        """
        if prediction.dim() != 4 or target.dim() != 4:
            raise ValueError("Input tensors must be 4-dimensional [B, C, H, W]")
        
        ssim_index = ssim(prediction, target) # Compute SSIM
        loss = 1 - ssim_index  # SSIM loss
        
        return loss.mean()  # Return mean loss over the batch