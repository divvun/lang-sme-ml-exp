import torch
import torch.nn as nn
from encode_input import load_corpus_chars
from encode_words import load_corpus_words, load_pos

def init_model(tokens, pos, device, is_gru, bidirectional, emb_dim=128, n_hidden=756, n_layers=2):
    
    tokens = load_corpus_words()
    pos = load_pos()
    model = RNN(tokens, pos, device, is_gru, bidirectional, emb_dim, n_hidden, n_layers)

    return model

def init_opt(model, lr=0.0001):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return opt
    

    
