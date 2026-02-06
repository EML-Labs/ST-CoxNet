import torch
import torch.nn.functional as F
from typing import List,Dict

def training_step(
    model, 
    rr_windows: torch.Tensor, 
    hrv_targets: torch.Tensor
) -> torch.Tensor:
    """
    rr_windows: [B, T, W]  -> RR windows
    hrv_targets : [B, T, num_metrics] -> HRV targets for different horizons
    Returns:
        loss: scalar tensor
    """
    # Get context embeddings from model
    c_seq = model(rr_windows)  # [B, T, context_dim]
    last_context = c_seq[:, -1, :]  # [B, context_dim]
    y_pred_1, y_pred_2, y_pred_4 = model.predictor(last_context)  # Each: [B, num_metrics]
    y_true_1, y_true_2, y_true_4 = hrv_targets[:, -1, :], hrv_targets[:, -2, :], hrv_targets[:, -3, :]  # Each: [B, num_metrics]
    loss_1 = F.mse_loss(y_pred_1, y_true_1)
    loss_2 = F.mse_loss(y_pred_2, y_true_2)
    loss_4 = F.mse_loss(y_pred_4, y_true_4)
    total_loss = loss_1 + loss_2 + loss_4
    return total_loss,loss_1,loss_2,loss_4
