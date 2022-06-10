import torch.nn as nn
# https://pytorch.org/vision/stable/models.html
import torchvision.models as models


class AllergyClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        convnext = models.convnext_large(pretrained=True)
        self.convnext = nn.Sequential(*list(convnext.children())[:])

    def forward(self, x):
        return self.convnext(x)
