import torch.nn as nn
from torchvision.models import convnext_tiny


class ConvNeXtBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = convnext_tiny(weights='DEFAULT')
        self.features = backbone.features

    def forward(self, x):
        return self.features(x)