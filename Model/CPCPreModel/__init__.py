import torch.nn as nn
import torch
from Model.AutoregressiveBlock import ARBlock
from Model.PredictionHead.HRVPredictor.MultiStepPredictor import MultiStepHRVPredictor
from Model.Encoder.RREncoder import Encoder
from Configs import LATENT_SIZE, CONTEXT_SIZE, NUMBER_OF_TARGETS_FOR_PREDICTION

class CPCPreModel(nn.Module):
    def __init__(self, num_targets=NUMBER_OF_TARGETS_FOR_PREDICTION):
        super().__init__()
        self.encoder = Encoder(latent_dim=LATENT_SIZE)
        self.context = ARBlock(latent_dim=LATENT_SIZE, context_dim=CONTEXT_SIZE)
        self.predictor = MultiStepHRVPredictor(context_dim=CONTEXT_SIZE, num_targets=num_targets)

    def forward(self, rr_windows: torch.Tensor) -> torch.Tensor:
        """
        rr_windows: [B, T, W] 
        Returns:
            c_seq: [B, T, CONTEXT_SIZE]
        """
        
        B, T, W = rr_windows.shape
        z_list = []

        for t in range(T):
            z_t = self.encoder(rr_windows[:, t, :])  # [B, LATENT_SIZE]
            z_list.append(z_t)

        z_seq = torch.stack(z_list, dim=1)  # [B, T, LATENT_SIZE]
        c_seq = self.context(z_seq)         # [B, T, CONTEXT_SIZE]
        return c_seq



