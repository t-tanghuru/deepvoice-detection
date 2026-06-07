import torch
import torch.nn as nn


class MFM(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class LCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), MFM(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, 3, padding=1), MFM(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 128, 3, padding=1), MFM(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64*4*4, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)
