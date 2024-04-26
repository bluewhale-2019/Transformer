import torch
from torch import nn 
from blocks.encoder_block import EncoderBlock
from blocks.decoder_block import DecoderBlock
from embedding import Transformer_Embedding

class Encoder(nn.Module):
    def __init__(self, encoder_vocab_size, max_len, d_model, hidden_layer, n_head, n_layer, drop_prob, device):
        super(Encoder, self).__init__()
        self.embedding  = Transformer_Embedding(vocab_size=encoder_vocab_size,
                                                d_model=d_model,
                                                max_len=max_len,
                                                drop_prob=drop_prob,
                                                device=device)
        self.layer = nn.ModuleList([EncoderBlock(d_model=d_model, hidden_layer=hidden_layer, n_head=n_head, drop_prob=drop_prob) for _ in range(n_layer)])
        
    def forward(self, x, mask):
        x = self.embedding(x)
        
        for layer in self.layer:
            x = layer.forward(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, decoder_vocab_size, max_len, d_model, hidden_layer, n_head, n_layer, drop_prob, device):
        super(Decoder, self).__init__()
        self.embedding  = Transformer_Embedding(vocab_size=decoder_vocab_size,
                                                d_model=d_model,
                                                max_len=max_len,
                                                drop_prob=drop_prob,
                                                device=device)
        self.layer = nn.ModuleList([DecoderBlock(d_model=d_model, hidden_layer=hidden_layer, n_head=n_head, drop_prob=drop_prob) for _ in range(n_layer)])
        self.linear = nn.Linear(d_model, decoder_vocab_size)
        
    def forward(self, dec, dec_mask, enc, enc_mask):
        dec = self.embedding(dec)
        
        for layer in self.layer:
            x = layer.forward(dec, enc, dec_mask, enc_mask)
        x = self.linear(x)
        return x

class Transformer(nn.Module):
    def __init__(self, source_padding_idx, target_padding_idx, target_start_idx, encoder_vocab_size, decoder_vocab_size,
                 max_len, d_model, hidden_layer, n_head, n_layer, drop_prob, device):
        super(Transformer, self).__init__()
        self.source_padding_idx = source_padding_idx
        self.target_padding_idx = target_padding_idx
        self.target_start_idx = target_start_idx
        self.device = device
        self.encoder = Encoder(decoder_vocab_size=decoder_vocab_size, 
                               d_model=d_model, 
                               max_len=max_len,
                               hidden_layer=hidden_layer,
                               n_head=n_head,
                               n_layer=n_layer,
                               drop_prob=drop_prob,
                               device=device)
        
        self.decoder = Decoder(encoder_vocab_size=encoder_vocab_size, 
                               d_model=d_model, 
                               max_len=max_len,
                               hidden_layer=hidden_layer,
                               n_head=n_head,
                               n_layer=n_layer,
                               drop_prob=drop_prob,
                               device=device)
        
    def forward(self, source, target):
        source_mask = self.get_source_mask(source)
        target_mask = self.get_target_mask(target)
        encoder_source = self.encoder.forward(source, source_mask)
        output = self.decoder.forward(target, encoder_source, target_mask, source_mask)
        return output
    
    def get_source_mask(self, source):
        source_mask = (source != self.source_padding_idx).unsqueeze(1).unsqueeze(2)
        return source_mask

    def get_target_mask(self, target):
        target_padding_mask = (target != self.target_padding_idx).unsqueeze(1).unsqueeze(3)
        target_len = target.shape[1]
        target_sub_mask = torch.tril(torch.ones(target_len, target_len)).type(torch.ByteTensor).to(self.device)
        target_mask = target_padding_mask & target_sub_mask
        return target_mask