import torch.nn as nn
import torch
from typing import Tuple
from Model.PredictionHead.HRVPredictor.SinglePredictor import HRVPredictor
from Configs import CONTEXT_SIZE, NUMBER_OF_TARGETS_FOR_PREDICTION, NUMBER_OF_PREDICTORS

class MultiStepHRVPredictor(nn.Module):
    def __init__(self, context_dim=CONTEXT_SIZE,num_heads:int =NUMBER_OF_PREDICTORS, num_targets=NUMBER_OF_TARGETS_FOR_PREDICTION):
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