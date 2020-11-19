import torch
import torch.nn as nn
from encode_input import load_corpus_chars

class RNN(nn.Module):

    def __init__(self, tokens, device, is_gru, bidirectional, use_embeddings, emb_dim=128, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.device = device
        self.bidirectional = bidirectional
        self.use_embeddings = use_embeddings
        self.emb_dim = emb_dim
        self.is_gru = is_gru

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        if use_embeddings:
            self.emb = nn.Embedding(len(self.chars), emb_dim)

            if is_gru:
                self.rnn = nn.GRU(emb_dim, n_hidden, n_layers, bidirectional=bidirectional, dropout=drop_prob, batch_first=True)
            else:
                self.rnn = nn.LSTM(emb_dim, n_hidden, n_layers, bidirectional=bidirectional, dropout=drop_prob, batch_first=True)
        else:
            if is_gru:
                self.rnn = nn.GRU(len(self.chars), n_hidden, n_layers, bidirectional=bidirectional, dropout=drop_prob, batch_first=True)
            else:
                self.rnn= nn.LSTM(len(self.chars), n_hidden, n_layers, bidirectional=bidirectional, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden*2 if bidirectional else n_hidden, len(self.chars))

    def forward(self, x, hidden):

        # self.lstm.flatten_parameters()
        if self.use_embeddings:
            x = self.emb(x)

        r_output, hidden = self.rnn(x, hidden)

        out = self.dropout(r_output)

        if not self.bidirectional:
            out = out.contiguous().view(-1, self.n_hidden)

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

def init_model(chars, device, is_gru, bidirectional, use_embeddings=True, emb_dim=128, n_hidden=756, n_layers=2):
    chars = load_corpus_chars()
    model = RNN(chars, device, is_gru, bidirectional, use_embeddings, emb_dim, n_hidden, n_layers)

    return model

def init_opt(model, lr=0.0001):
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    return opt
