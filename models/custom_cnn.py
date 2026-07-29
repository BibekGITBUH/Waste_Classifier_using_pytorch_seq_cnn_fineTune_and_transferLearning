"""
A custom CNN built from scratch (no pretrained weights) as the baseline
model to compare against transfer learning later.

Architecture: 4 conv blocks (Conv -> BatchNorm -> ReLU -> MaxPool) with
increasing channel depth, followed by global average pooling and a small
fully-connected classifier head. BatchNorm + Dropout are used to reduce
overfitting, since this network has no pretrained prior to lean on.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),     # 224 -> 112
            ConvBlock(32, 64),    # 112 -> 56
            ConvBlock(64, 128),   # 56  -> 28
            ConvBlock(128, 256),  # 28  -> 14
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> (batch, 256, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CustomCNN(num_classes=5)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect (2, 5)
    print(f"Trainable parameters: {count_parameters(model):,}")
