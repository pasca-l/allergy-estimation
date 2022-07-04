import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
# https://pytorch.org/vision/stable/models.html
import torchvision.models as models


class AllergyClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        convnext = models.convnext_tiny(pretrained=True)
        self.convnext = nn.Sequential(*list(convnext.children())[:-1])
        self.seq = nn.Sequential(
            LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=768, out_features=101, bias=True)
        )

    def forward(self, x):
        x = self.convnext(x)
        x = self.seq(x)
        return x


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = nnf.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


# class AllergyLinear(nn.Module):
#     def __init__(self, weight_file='../data/meta/weights.csv'):
#         super().__init__()
#         weights = np.loadtxt(weight_file, delimiter=',', skiprows=1, 
#                              usecols=range(1, 28), dtype='float32')
#         self.weight = nn.Parameter(torch.as_tensor(weights.T), 
#                                    requires_grad=False)

#     def forward(self, x):
#         x = nnf.linear(x, self.weight)
#         return x