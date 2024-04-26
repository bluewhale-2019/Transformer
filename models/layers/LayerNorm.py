import torch
from torch import nn 

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-9):
        super(LayerNorm, self).__init__()
        self.weight = nn.parameter.Parameter(torch.ones(d_model))
        self.bias = nn.parameter.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x:torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.weight * mean + self.bias
        return out
        