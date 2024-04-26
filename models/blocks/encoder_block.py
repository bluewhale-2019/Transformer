import torch
from torch import nn 
from models.layers.LayerNorm import LayerNorm
from models.layers.MH_Attention import MultiHeadAttention
from models.layers.Position_wise_FFN import PositionwiseFeedForward

class EncoderBlock(nn.module):
    def __init__(self, d_model, hidden_layer, n_head, drop_prob):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden_layer=hidden_layer, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
    def forward(self, x, mask):
        # self_attention
        sub_x = x
        x = self.attention.forward(q=x, k=x, v=x, mask=mask)
        
        # add and norm
        x = self.dropout1(x)
        x = self.norm1(x + sub_x)
        
        # Position_wise feed forward network
        sub_x = x
        x = self.ffn.forward(x=x)
        
        # add and norm
        x = self.dropout2(x)
        x = self.norm2(x + sub_x)
        return x
        