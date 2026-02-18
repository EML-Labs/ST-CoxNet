import torch.nn as nn
import torch
from typing import Tuple
from Model.PredictionHead.HRVPredictor.SinglePredictor import HRVPredictor

class MultiStepHRVPredictor(nn.Module):
    def __init__(self, context_dim:int, num_heads:int, num_targets:int):
        super().__init__()
        self.predictors = nn.ModuleList(
            [HRVPredictor(context_dim, num_targets) for _ in range(num_heads)]
        )
    def forward(self, c_t:torch.Tensor)->Tuple[torch.Tensor,...]:
        """
        c_t: [B, context_dim]
        returns:
            Tuple of predictions from each predictor
        """
        predictions = [predictor(c_t) for predictor in self.predictors]
        return tuple(predictions)