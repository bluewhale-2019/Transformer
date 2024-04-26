import torch
from torch import nn 

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = torch.softmax(dim=-1)
        
    def forward(self, q, k, v, mask=None, eps=1e-9):
        # input size: [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        
        score = torch.matmul(q, torch.transpose(k, 2, 3)) / torch.sqrt(d_tensor)
        if mask is not None:
            score = score.masked_fill(mask==0, 10000)
        score = torch.softmax(score)
        attention_qkv = torch.matmul(torch.softmax(score), v)
        return attention_qkv, score
        
        