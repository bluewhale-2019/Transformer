import torch
from torch import nn

class Positional_Encoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(Positional_Encoding, self).__init__()
        
        self.encoder = torch.zeros((max_len, d_model), device=device)
        self.encoder.requires_grad = False
        
        Position = torch.arange(0, max_len, device=device)
        Position = Position.float().unsqueeze(dim=1)
        
        _i = torch.arange(0, d_model, device=device).float()
        self.encoder[:,0::2] = torch.sin(Position / (10000 ** (2 * _i / d_model)))
        self.encoder[:,1::2] = torch.cos(Position / (10000 ** (2 * _i / d_model)))
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        return self.encoder[:seq_len, :]

class Token_Embedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(Token_Embedding, self).__init__(vocab_size, d_model, padding_idx=1)
    
class Transformer_Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(Transformer_Embedding, self).__init__()
        self.pos_embed = Positional_Encoding(d_model=d_model, max_len=max_len,device=device)
        self.tok_embed = Token_Embedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = nn.Dropout(p=drop_prob)
        
    def forward(self, x):
        tok_embed = self.tok_embed(x)
        pos_embed = self.pos_embed(x)
        return self.dropout(tok_embed+pos_embed)