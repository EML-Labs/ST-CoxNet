from Metadata import OptimizerConfig, OptimizerType
import torch.optim as optim

class Optimizer:
    def __init__(self, config: OptimizerConfig, model_parameters):
        self.name = config.name
        self.lr = config.lr
        if self.name == OptimizerType.ADAMW:
            self._optimizer = optim.AdamW(model_parameters, lr=self.lr)
        elif self.name == OptimizerType.ADAM:
            self._optimizer = optim.Adam(model_parameters, lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.name}")
        
    @property
    def optimizer(self):
        return self._optimizer