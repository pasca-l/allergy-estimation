import torch.nn as nn
# https://pytorch.org/vision/stable/models.html
import torchvision.models as models

class AllergyClassifierModelFineTune():
    def __init__(self):
        self.arch = models.convnext_large(pretrained=True)
