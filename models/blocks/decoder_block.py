import torch
from torch import nn 
from models.layers.LayerNorm import LayerNorm
from models.layers.MH_Attention import MultiHeadAttention
from models.layers.Position_wise_FFN import PositionwiseFeedForward

class DecoderBlock(nn.Module):
    def __init__(self, d_model, hidden_layer, n_head, drop_prob):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        self.en_de_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden_layer=hidden_layer, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)
        
    def forward(self, dec, enc, dec_mask, enc_mask):
        # self_attention
        sub_x = dec
        x = self.attention.forward(q=dec, k=dec, v=dec, mask=dec_mask)
        
        # add and norm
        x = self.dropout1(x)
        x = self.norm1(x + sub_x)
        
        if enc is not None:
            sub_x = x
            x = self.attention.forward(q=x, k=enc, v=enc, mask=enc_mask)
            
            # add and norm
            x = self.dropout2(x)
            x = self.norm2(x + sub_x)
            
        # Position_wise feed forward network
        sub_x = x
        x = self.ffn.forward(x=x)
        
        # add and norm
        x = self.dropout3(x)
        x = self.norm3(x + sub_x)
        return x