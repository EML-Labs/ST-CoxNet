import torch

class Device:
    def __init__(self,device:str = "cpu"):
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    @property
    def device(self):
        return self._device
