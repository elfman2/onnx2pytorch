import torch
from torch import nn


class CumSum(nn.Module):
    def forward(self, x:torch.Tensor, dim:torch.Tensor):
        return torch.cumsum(x, dim.item())
