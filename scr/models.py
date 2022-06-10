import torch.nn as nn


class AllergyClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self(x)
