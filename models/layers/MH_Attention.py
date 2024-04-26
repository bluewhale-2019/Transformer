import torch
from torch import nn 
import copy

from SDP_Attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        q, k ,v = torch.matmul(self.w_q, q), torch.matmul(self.w_k, k), torch.matmul(self.w_v, v)
        
        # split by numbers of heads
        # q, k, v = torch.split(q, split_size_or_sections=self.n_head), torch.split(k, split_size_or_sections=self.n_head), torch.split(v, split_size_or_sections=self.n_head)
        q, k, v = self.split(q), self.split(k), self.split(v)
        
        out, attention_qkv = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out
    
    def split(self, x):
        """
        Args:
            x (_type_): torch.tensor
        input_size:[batch_size, length, d_model]
        output_size:[batch_size, head, length, d_model]
        """
        batch_size, length, d_model = x.size()
        
        d_tensor = d_model // self.n_head
        x = x.view(batch_size, length, self.n_head, d_tensor)
        x = torch.transpose(x, 1, 2)
        
        return x
    
    def concat(self, x):
        """
        Args:
            x (_type_): torch.tensor
        input_size:[batch_size, length, d_model]
        output_size:[batch_size, head, length, d_model]
        """
        batch_size, head, length, d_tensor = x.size() 
        d_model = head * d_tensor
        
        x = torch.transpose(x, 1, 2)
        x = copy.deepcopy(x)
        x = x.view(batch_size, length, d_model)
        return x