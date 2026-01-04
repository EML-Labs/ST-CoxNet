import torch.nn as nn

import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, emb_dim=16, dropout=0.0):
        super().__init__()

        # 1. Expand latent vector to match the Encoder's bottleneck spatial size (16 channels x 8 x 8)
        # Note: 16 * 8 * 8 = 1024 features
        self.linear = nn.Linear(emb_dim, 16 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (16, 8, 8))   # Reshape to [B, 16, 8, 8]

        # Block 1: Upsample 8x8 -> 16x16
        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # [B, 16, 16, 16]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B, 32, 16, 16]
            nn.BatchNorm2d(32),                           # Added BatchNorm for training stability
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # Block 2: Upsample 16x16 -> 32x32
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # [B, 32, 32, 32]
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # [B, 16, 32, 32]
            nn.BatchNorm2d(16),                           # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # Block 3: Final projection to 1 Channel (Original Image)
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),   # [B, 1, 32, 32]
            nn.Sigmoid()  # Forces output to [0, 1] range
        )

    def forward(self, x):
        x = self.linear(x)      # [B, 1024]
        x = self.unflatten(x)   # [B, 16, 8, 8]
        
        x = self.block1(x)      # [B, 32, 16, 16]
        x = self.block2(x)      # [B, 16, 32, 32]
        x = self.block3(x)      # [B, 1, 32, 32]
        
        return x