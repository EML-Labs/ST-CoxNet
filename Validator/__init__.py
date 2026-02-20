import torch
from typing import List,Dict,Tuple


class Validator:
    loss_weights = []
    def __init__(self, model: torch.nn.Module, device: torch.device, loss: torch.nn.Module, number_of_predictors: int):
        self.model = model
        self.device = device
        self.loss_weights = [1.0 for _ in range(number_of_predictors)]
        self.loss = loss
        self.predictions = []
        self.targets = []

    def validation_step(
        self, 
        rr_windows: torch.Tensor, 
        hrv_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        rr_windows: [B, T, W]  -> RR windows
        hrv_targets : [B, T, num_metrics] -> HRV targets for different horizons
        Returns:
            total_loss: scalar tensor
            losses: tuple of individual losses
        """
        losses = []
        # Get context embeddings from model
        c_seq = self.model(rr_windows)  # [B, T, context_dim]
        last_context = c_seq[:, -1, :]  # [B, context_dim]
        total_loss = 0.0

        y_preds = self.model.predictor(last_context)  # Each: [B, num_metrics]
        self.predictions.append([y.cpu().detach() for y in y_preds])
        self.targets.append(hrv_targets.cpu().detach())
        for idx,y_pred in enumerate(y_preds):
            loss = self.loss(y_pred, hrv_targets[:, idx, :])
            losses.append(loss)
            total_loss += self.loss_weights[idx] * loss
    
        return total_loss,tuple(losses)
    

    def validation_epoch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        losses = [0.0 for _ in self.loss_weights]
        num_batches = 0

        self.predictions = []
        self.targets = []
        for rr_windows, hrv_targets, _ in dataloader:
            rr_windows = rr_windows.to(self.device)  # [B, T, W]
            hrv_targets = hrv_targets.to(self.device)  # [B, T, num_metrics]

            loss, batch_losses = self.validation_step(rr_windows, hrv_targets)

            total_loss += loss.item()
            for idx, l in enumerate(batch_losses):
                losses[idx] += l.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_losses = [l / num_batches for l in losses]
        return avg_loss, tuple(avg_losses)
