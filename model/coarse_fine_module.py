import torch
import torch.nn as nn


class CoarseFineModule(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.coarse_branch = nn.Conv2d(channels, channels, 3, padding=1)
        self.fine_branch = nn.Conv2d(channels, channels, 1)

        self.fusion = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        coarse = self.coarse_branch(x)
        fine = self.fine_branch(x)

        fused = torch.cat([coarse, fine], dim=1)
        fused = self.fusion(fused)

        return fused