from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

class Dataloader:
    def __init__(self, ext, token_encoder, token_decoder, init_token, eos_token) -> None:
        self.ext = ext
        self.token_encoder = token_encoder
        self.token_decoder = token_decoder
        self.init_token = init_token
        self.eos_token = eos_token
        
    def preprocess(self):
        if self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.token_encoder, init_token=self.init_token, eos_token=self.eos_token, lower=True, batch_first=True)
            self.target = Field(tokenize=self.token_decoder, init_token=self.init_token, eos_token=self.eos_token, lower=True, batch_first=True)
        elif self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.token_decoder, init_token=self.init_token, eos_token=self.eos_token, lower=True, batch_first=True)
            self.target = Field(tokenize=self.token_encoder, init_token=self.init_token, eos_token=self.eos_token, lower=True, batch_first=True) 
            
        train_data, vali_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, vali_data, test_data
    
    def bulid_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)
        
    def iterator(self, train_data, vali_data, test_data, batch_size, device):
        train_iterator, vali_iterator, test_iterator = BucketIterator.splits((train_data, vali_data, test_data), batch_sizes=batch_size,device=device)
        return train_iterator, vali_iterator, test_iterator