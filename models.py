import torch
import torch.nn as nn
from encode_input import load_corpus_chars
from encode_words import load_corpus_words
class RNN(nn.Module):

    def __init__(self, tokens, device, is_gru, bidirectional, emb_dim=128, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.device = device
        self.bidirectional = bidirectional
        # self.use_embeddings = use_embeddings
        self.emb_dim = emb_dim
        self.is_gru = is_gru
        self.tokens = tokens
        # self.int2item = dict(enumerate(self.tokens))
        # self.item2int = {ch: ii for ii, ch in self.int2item.items()}
        # self.pos = pos
        # tokens = tokens
        # self.int2word = dict(enumerate(self.pos))
        # self.word2int = 
        # if use_embeddings:
        self.emb = nn.Embedding(len(self.tokens), emb_dim)
            # self.emb_pos = nn.Embedding(len(self.pos), emb_dim)
        if is_gru:
            self.rnn = nn.GRU(emb_dim, n_hidden, n_layers, bidirectional=bidirectional, dropout=drop_prob, batch_first=True)
        else:
            self.rnn = nn.LSTM(emb_dim, n_hidden, n_layers, bidirectional=bidirectional, dropout=drop_prob, batch_first=True)
        # else:
        #     if is_gru:
        #         self.rnn = nn.GRU(len(tokens), n_hidden, n_layers, bidirectional=bidirectional, dropout=drop_prob, batch_first=True)
        #     else:
        #         self.rnn= nn.LSTM(len(tokens), n_hidden, n_layers, bidirectional=bidirectional, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden*2 if bidirectional else n_hidden, len(tokens))

    def forward(self, x, hidden):

        # x_pos = self.pos
        # x = torch.cat((x, x_pos), dim=1)
        # self.lstm.flatten_parameters()
        # if self.use_embeddings:
        x = self.emb(x)

        if self.is_gru:
            r_output, hidden = self.rnn(x)
        else:
            r_output, hidden = self.rnn(x, hidden)

        out = self.dropout(r_output)

        # if not self.bidirectional:
            # out = out.contiguous().view(-1, self.n_hidden)

        out = self.fc(out)

        return out, hidden


    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data

        if self.is_gru:
            if self.bidirectional:
                hidden = (weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_().to(self.device))
            else:
                hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device))
        else:
            if self.bidirectional:
                hidden = (weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_().to(self.device),
                    weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_().to(self.device))
            else:
                hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device))

        return hidden

def init_model(tokens, device, is_gru, bidirectional, emb_dim=128, n_hidden=756, n_layers=2):
    
    tokens = load_corpus_words()
    
    model = RNN(tokens, device, is_gru, bidirectional, emb_dim, n_hidden, n_layers)

    return model

def init_opt(model, lr=0.0001):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return opt
    

    
