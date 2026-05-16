import torch
import torch.nn as nn

from model.convnext_backbone import ConvNeXtBackbone
from model.coarse_fine_module import CoarseFineModule
from model.tri_path_fusion import TriPathFusion
from model.mamba_block import MambaBlock


class HSRDSMamba(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.backbone = ConvNeXtBackbone()

        self.coarse_fine = CoarseFineModule(768)
        self.tri_path = TriPathFusion(768)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.mamba = MambaBlock(768)

        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.backbone(x)

        x = self.coarse_fine(x)
        x = self.tri_path(x)

        x = self.pool(x).flatten(1)

        x = self.mamba(x)

        x = self.classifier(x)

        return x