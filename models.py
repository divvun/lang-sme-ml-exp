import torch.nn as nn

class LSTM(nn.Module):
    
    def __init__(self, tokens, device, bidirectional, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001, ):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.device = device
        self.bidirectional = bidirectional
        
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, bidirectional=bidirectional,
                            dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden*2 if bidirectional else n_hidden,len(self.chars))
      
    
    def forward(self, x, hidden):
        
        
        r_output, hidden = self.lstm(x, hidden)
        
        out = self.dropout(r_output)
        
        if not self.bidirectional:

            out = out.contiguous().view(-1, self.n_hidden)
        
        
        out = self.fc(out)
        
    
        return out, hidden
    
    
    def init_hidden(self, batch_size):
       
        weight = next(self.parameters()).data
        
       
        if self.bidirectional:

            hidden = (weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_().to(self.device),
                weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_().to(self.device))
        else:
             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device))

        return hidden

