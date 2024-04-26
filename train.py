import torch.utils
from configs import get_best_gpu, get_args
from utils.dataloader import Dataloader
from utils.tokenizer import Tokenizer
from utils.bleu import idx_to_word, get_bleu
from models.transformer import Transformer
import torch
from torch import nn, optim
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import math
import os
import numpy as np

args = get_args()
device = torch.device("cpu")
if torch.cuda.is_available():
    device = get_best_gpu()
    # hyper_parameter
epoch = args.epoch
lr = args.lr
weight_decay = args.weight_decay
epsilon_ls = args.epsilon_ls     
batch_size = args.batch_size 
clip = args.clip


#load dataset
tokenizer = Tokenizer()
load_data = Dataloader(ext=('.en', '.de'), token_encoder=tokenizer.tokenizer_encoder, token_decoder=tokenizer.tokenizer_decoder, init_token='<sos>', eos_token='<eos>')
train_data, vali_data, test_data = load_data.preprocess()
load_data.bulid_vocab(train_data=train_data, min_freq=2)
train_iter, vali_iter, test_iter = load_data.iterator(train_data, vali_data, test_data, batch_size=args.batch_size, device=device)
source_padding_idx = load_data.source.vocab.stoi['<pad>']
target_padding_idx = load_data.target.vocab.stoi['<pad>']
target_start_idx = load_data.target.vocab.stoi['<sos>']

encoder_vocab_size = len(load_data.source.vocab)
decoder_vocab_size = len(load_data.source.vocab)

writer_train = SummaryWriter('./logs/train')
writer_vali = SummaryWriter('./logs/vali')


def init_weight(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.normal_(m.weight.data)
        
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for idx, batch in enumerate(iterator):
        source = batch.source
        target = batch.target
        
        optimizer.zero_grad()
        output = model(source, target[:,:-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        target = target[:,1:].contiguous().view(-1)
        
        loss = criterion(output_reshape, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            source = batch.source
            target = batch.target
            
            output = model(source, target[:,:-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            target = target[:,1:].contiguous().view(-1) 
            
            loss = criterion(output_reshape, target)
            epoch_loss += loss.item()
            
            total_bleu = []
            for i in range(batch_size):
                try:
                    target_words = idx_to_word(batch.trg[i], load_data.target.vocab)
                    output_words = output[i].max(dim=1)[1]
                    output_words = idx_to_word(output_words, load_data.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=target_words.split())
                    total_bleu.append(bleu)
                except:
                    pass
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu
                


if __name__ == '__main__':
    Transformer_Model = Transformer(source_padding_idx=source_padding_idx,
                                    target_padding_idx=target_padding_idx,
                                    target_start_idx=target_start_idx,
                                    encoder_vocab_size=encoder_vocab_size,
                                    decoder_vocab_size=decoder_vocab_size,
                                    d_model=args.d_model,
                                    max_len=args.max_len,
                                    hidden_layer=args.hidden_layer,
                                    n_head=args.n_head,
                                    n_layer=args.n_layer,
                                    drop_prob=args.drop_prob,
                                    device=device)
    Transformer_Model.apply(init_weight)
    
    optimizer = Adam(params=Transformer_Model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay, eps=epsilon_ls)
    criterion = nn.CrossEntropyLoss(ignore_index=source_padding_idx)
    
    train_losses, vali_losses, bleus = [], [], []
    best_loss = float('inf')
    for i in range(epoch):
        train_loss = train(Transformer_Model, train_iter, optimizer, criterion, clip)
        vali_loss, bleu = evaluate(Transformer_Model, vali_iter, criterion)
        
        train_losses.append(train_loss)
        vali_losses.append(vali_loss)
        bleus.append(bleu)
        
        if vali_loss < best_loss:
            best_loss = vali_loss
            torch.save(Transformer_Model.state_dict(), './checkpoints/model_{0}.pf'.format(vali_loss))
        
        writer_train.add_scalar('train_loss', train_loss, i)
        writer_vali.add_scalar('vali_loss', vali_loss, i)  
        print('Epoch: {0}----------Train Loss: {1} Vali_loss: {2} BLEU Score: {3}'.format(i+1, train_loss, vali_loss, bleu))
    
    total_losses = np.concatenate((train_losses, vali_losses), axis=1)
    np.save('./results/total_losses.npy', total_losses)
    np.save('./results/BLEU.npy', bleus)
    writer_train.close()
    writer_vali.close()