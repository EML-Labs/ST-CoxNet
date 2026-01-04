import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, emb_dim=16, dropout=0.1):
        super().__init__()

        # Block 1 (32 → 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        # Block 2 (32 → 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2)  # 32 → 16

        # Block 3 (16 → 16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Block 4 (16 → 16)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.pool2 = nn.MaxPool2d(2)  # 16 → 8

        # Bottleneck conv: reduce channels 64 → 16
        self.conv_bottleneck = nn.Conv2d(64, 16, kernel_size=1)  # [B,16,8,8]

        # Fully connected to emb_dim
        self.flatten = nn.AdaptiveAvgPool2d((1, 1))  # [B,16,8,8] -> [B,16,1,1]
        self.fc = nn.Sequential(
            nn.Linear(16, emb_dim), # compress to emb_dim (e.g., 64)
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.conv1(x)   # [B,8,32,32]
        x = self.conv2(x)   # [B,16,32,32]
        x = self.pool1(x)   # [B,16,16,16]

        x = self.conv3(x)   # [B,32,16,16]
        x = self.conv4(x)   # [B,64,16,16]
        x = self.pool2(x)   # [B,64,8,8]

        x = self.conv_bottleneck(x)  # [B,16,8,8]
        x = self.flatten(x)           # [B,16*8*8 = 1024]
        x = self.fc(x)                # [B, emb_dim]
        return x
