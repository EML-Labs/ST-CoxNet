from torch.nn import MSELoss
from Loss.SSIMLoss import SSIMLoss
from Loss.SymmetricLoss import SymmetricLoss
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self,alpha:float=0.35,beta:float=0.45,gamma:float=0.2):
        super(ReconstructionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse_loss = MSELoss()
        self.ssim_loss = SSIMLoss()
        self.symmetric_loss = SymmetricLoss()

    def forward(self, prediction, target):
        mse = self.mse_loss(prediction, target)
        ssim = self.ssim_loss(prediction, target)
        sym = self.symmetric_loss(prediction)
        total_loss = self.alpha * mse + self.beta * ssim + self.gamma * sym
        return total_loss,{'mse_loss': mse, 'ssim_loss': ssim, 'symmetric_loss': sym}