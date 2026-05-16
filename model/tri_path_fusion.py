import torch
import torch.nn as nn


class TriPathFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.path1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.path2 = nn.Conv2d(channels, channels, 5, padding=2)
        self.path3 = nn.Conv2d(channels, channels, 1)

        self.fusion = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)

        out = torch.cat([p1, p2, p3], dim=1)
        out = self.fusion(out)

        return out