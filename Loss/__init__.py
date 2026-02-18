from Metadata import LossConfig,LossType
import torch.nn as nn

class Loss:
    def __init__(self, config: LossConfig):
        self.name = config.name
        if self.name == LossType.MSE:
            self._loss_fn = nn.MSELoss()
        elif self.name == LossType.MAE:
            self._loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {self.name}")
        
    @property
    def loss_fn(self):        
        return self._loss_fn

