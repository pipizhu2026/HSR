import torch
import torch.nn as nn


class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x

        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)

        return x + residual