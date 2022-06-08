import torch.nn as nn
# https://pytorch.org/vision/stable/models.html
import torchvision.models as models


class AllergyClassifierFinetuneModel():
    def __init__(self):
        self.model_arch = models.convnext_large(pretrained=True)

    def forward(self, x):
        return self.model_arch(x)
