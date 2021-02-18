import os
import argparse
import typing
import csv
import sys
import signal
from functools import lru_cache

import torch
import torch.nn as nn
import numpy as np 

from torch.utils.data import DataLoader, Dataset

class InputPaths(typing.NamedTuple):
    # Model output directory
    model_dir: str
    # Checkpoints, etc. Does not include model.
    output_dir: str
    # Training input data directory
    train_dir: str
    # checkpont directory
    checkpoint_dir: str

    @property
    def model_path(self):
        return os.path.join(self.model_dir, "model.pth")
    
    @property
    def checkpoint_path(self):
        return os.path.join(self.checkpoint_dir, "checkpoint.pth")

    @property
    def pos_path(self):
        return os.path.join(self.train_dir, "encoded-pos.txt")

    @lru_cache(maxsize=None)
    def load_pos(self):
        with open(self.pos_path, 'r', encoding = "utf-8") as f:
            return f.read().split('\n')

    @property
    def corpus_words_path(self):
        return os.path.join(self.train_dir, "corpus-words.txt")
    
    @lru_cache(maxsize=None)
    def load_corpus_words(self):
        with open(self.corpus_words_path, 'r', encoding = "utf-8") as f:
            return f.read().split('\n')

    @property
    def train_words_path(self):
        return os.path.join(self.train_dir, "train_words_enc.csv")

    @lru_cache(maxsize=None)
    def train_words(self):
        with open(self.train_words_path, 'r') as f:
            train = list(csv.reader(f))
            x_train = [int(i[0]) for i in train]
            pos_x_train = [int(i[1]) for i in train]
            y_train = [int(i[2]) for i in train]
            pos_y_train = [int(i[3]) for i in train]

        return (x_train, pos_x_train, y_train, pos_y_train)
       
    @property
    def val_words_path(self):
        return os.path.join(self.train_dir, "val_words_enc.csv")

    @lru_cache(maxsize=None)
    def val_words(self):
        with open(self.val_words_path, 'r') as f:
            train = list(csv.reader(f))
            x_train = [int(i[0]) for i in train]
            pos_x_train = [int(i[1]) for i in train]
            y_train = [int(i[2]) for i in train]
            pos_y_train = [int(i[3]) for i in train]

        return (x_train, pos_x_train, y_train, pos_y_train)


def get_batch(inputs, b_size):
    (x, x_pos, y, y_pos) = inputs

    data = Data(x, y)
    data_pos = Data(x_pos, y_pos)

    batched = DataLoader(dataset=data, batch_size=b_size, drop_last=True, shuffle=False)
    batched_pos = DataLoader(dataset=data_pos, batch_size=b_size, drop_last=True, shuffle=False)

    return (batched, batched_pos)

class Data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__(self):
        return len(self.X_data)

class Hyperparams(typing.NamedTuple):
    epochs: int
    batch_size: int
    lr: int = 0.0001
    bidirectional: bool = False
    emb_dim: int = 128
    is_gru: bool = False
    n_hidden: int = 756
    n_layers: int = 2

class RNN(nn.Module):
    def __init__(self, tokens, pos, device, params: Hyperparams, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = params.n_layers
        self.n_hidden = params.n_hidden
        self.lr = params.lr
        self.device = device
        self.bidirectional = params.bidirectional
        # self.use_embeddings = use_embeddings
        self.emb_dim = params.emb_dim
        self.is_gru = params.is_gru
        self.tokens = tokens
        # self.int2item = dict(enumerate(self.tokens))
        # self.item2int = {ch: ii for ii, ch in self.int2item.items()}
        self.pos = pos
        # tokens = tokens
        # self.int2word = dict(enumerate(self.pos))
        # self.word2int = 
        # if use_embeddings:
        self.emb = nn.Embedding(len(self.tokens), params.emb_dim)
        self.emb_pos = nn.Embedding(len(self.pos), params.emb_dim)
        if params.is_gru:
            self.rnn = nn.GRU(params.emb_dim, params.n_hidden, params.n_layers, bidirectional=params.bidirectional, dropout=drop_prob, batch_first=True)
        else:
            self.rnn = nn.LSTM(params.emb_dim, params.n_hidden, params.n_layers, bidirectional=params.bidirectional, dropout=drop_prob, batch_first=True)
        # else:
        #     if is_gru:
        #         self.rnn = nn.GRU(len(tokens), n_hidden, n_layers, bidirectional=bidirectional, dropout=drop_prob, batch_first=True)
        #     else:
        #         self.rnn= nn.LSTM(len(tokens), n_hidden, n_layers, bidirectional=bidirectional, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(params.n_hidden * 2 if params.bidirectional else params.n_hidden, len(tokens))

    def forward(self, t, p, hidden):
        x = self.emb(t)
        x_pos = self.emb_pos(p)
        # print(x.shape)
        # concat two emb layers
        x = torch.cat((x, x_pos), 0)
        # self.lstm.flatten_parameters()
        # if self.use_embeddings:
        x = x.view(x.size(0), 1, x.size(1))
        # print(x.shape)
        if self.is_gru:
            r_output, hidden = self.rnn(x)
        else:
            r_output, hidden = self.rnn(x)

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


def save_checkpoint(model, opt, epoch, lr, batch_size, model_name):
    torch.save({
        'model': model.state_dict(),
        'is_gru': model.is_gru,
        'bidirectional': model.bidirectional,
        'emb_dim': model.emb_dim,
        'n_hidden': model.n_hidden,
        'n_layers': model.n_layers,
        'drop_prob': model.drop_prob,
        'lr' : lr,
        'opt': opt.state_dict(),
        'epoch': epoch,
        'batch_size': batch_size,
    }, model_name)

def start_training(device, params: Hyperparams, paths: InputPaths):
    tokens = paths.load_corpus_words()
    pos = paths.load_pos()

    model = RNN(tokens, pos, device, params)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params.lr)

    if os.path.exists(paths.checkpoint_path):
        checkpoint = torch.load(paths.checkpoint_path, map_location=torch.device(device))
        starting_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        print(f"\nResuming model training from epoch {starting_epoch} ...")
    else:
        starting_epoch = 0
        print(f"\nStarting to train model '{paths.model_path}'...")

    run_training_loop(model, opt, device, starting_epoch, params, paths)

def run_training_loop(model, opt, device, starting_epoch, params: Hyperparams, paths: InputPaths):
    print("Model's architecture: \n", model)
    print(f"Training with batch size: {params.batch_size}")

    clip = 5
    criterion = nn.CrossEntropyLoss()

    #train mode
    model.train()

    batch_size = params.batch_size
    
    for e in range(starting_epoch, params.epochs):
        print(f"Epoch: {e}")
        # initialize hidden state
        print("init hidden")
        h = model.init_hidden(batch_size)
      
        train_words_inputs = paths.train_words()
        total_inputs = len(train_words_inputs[0])
        print(f"Total inputs: {total_inputs}")

        print("Get batch")
        tok, pos = get_batch(train_words_inputs, batch_size)
        print("get batch finished")

        for (n, ((tok_x, tok_y), (pos_x, pos_y))) in enumerate(zip(tok, pos)):
            inputs_tok, targets_tok = tok_x.to(device), tok_y.to(device)
            inputs_pos, _targets_pos = pos_x.to(device), pos_y.to(device)

            # Creating a separate variables for the hidden state
            if not model.is_gru:
                h = tuple([each.data for each in h])
    
            model.zero_grad()
        
            output, h = model(inputs_tok, inputs_pos, h)
            # some reshaping for output needed
         
            output = output.view(batch_size, -1)
            loss = criterion(output, targets_tok)
            print(f"[{n+1}/{total_inputs}] {loss}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        print("val_h init hidden")
        val_h = model.init_hidden(batch_size)
        val_losses = []

        # move to eval mode
        model.eval()

        val_words_inputs = paths.val_words()
        total_val_inputs = len(val_words_inputs[0])
        tok, pos = get_batch(val_words_inputs, batch_size)

        for (n, ((tok_x, tok_y), (pos_x, pos_y))) in enumerate(zip(tok, pos)):
            inputs_tok, targets_tok = tok_x.to(device), tok_y.to(device)
            inputs_pos, _targets_pos = pos_x.to(device), pos_y.to(device)

            if not model.is_gru:
                val_h = tuple([each.data for each in val_h])

            output, val_h = model(inputs_tok, inputs_pos, val_h)

            output = output.view(batch_size, -1)
            val_loss = criterion(output, targets_tok)
            print(f"[{n+1}/{total_val_inputs}] {val_loss}")

            val_losses.append(val_loss.item())

        print("training")
        model.train() # reset to train

        print("saving checkpoint")
        save_checkpoint(model, opt, e, params.lr, batch_size, paths.checkpoint_path)

        # return current_params
        print("Epoch: {}...".format(e+1),
            "Loss: {:.4f}...".format(loss.item()),
            "Valid loss: {:.4f}".format(np.mean(val_losses)))

    print("Saving final model")
    with open(paths.model_path, 'wb') as f:
        torch.save(model.state_dict(), f)

def main():
    def sigterm_handler(signum, frame):
        if signum == signal.SIGTERM:
            print("SIGTERM received")
            print("TODO: save checkpoint")
            sys.exit(1)
        else:
            print(f"Signal received: {signum}")

    signal.signal(signal.SIGTERM, sigterm_handler)

    # AWS-required fields
    model_dir = os.environ.get('SM_MODEL_DIR', None)
    output_data_dir = os.environ.get('SM_OUTPUT_DATA_DIR', None)
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', None)
    checkpoint_dir = None if model_dir is None else "/opt/ml/checkpoints"

    p = argparse.ArgumentParser(description="parameters for training the model")
    p.add_argument('--model-dir', type=str, default=model_dir, required=True if model_dir is None else False)
    p.add_argument('--output-data-dir', type=str, default=output_data_dir, required=True if output_data_dir is None else False)
    p.add_argument('--train-dir', type=str, default=train_dir, required=True if train_dir is None else False)
    p.add_argument('--checkpoint-dir', type=str, default=checkpoint_dir)
    p.add_argument("--epochs", type=int, help='num epochs to train over', required=True)
    p.add_argument("--batch-size", type=int, help="mini_batch size", required=True)
    p.add_argument("--lr", type=float, help="learning rate: default 0.0001", default=0.0001)
    p.add_argument("--bidirectional", help="whether a model is bidirectonal: default False", action='store_true')
    p.add_argument("--emb-dim", type=int, help="size of emb_dim", default=128)
    p.add_argument("--is-gru", help="whether to use GRU or LSTM model", action='store_true')
    p.add_argument("--n-hidden", type=int, help="size of hidden layer", default=756)
    p.add_argument("--n-layers", type=int, help="num of layers in the model", default=2)
    p.add_argument("--device", type=str, help="cuda or cpu", default="cuda:0")

    args = p.parse_args()

    if args.device.startswith("cuda"):
        import torch.cuda
        if not torch.cuda.is_available():
            print("CUDA not found to be available.")
            sys.exit(1)

    params = Hyperparams(
        epochs = args.epochs,
        batch_size = args.batch_size,
        lr = args.lr,
        bidirectional = args.bidirectional,
        emb_dim = args.emb_dim,
        is_gru = args.is_gru,
        n_hidden = args.n_hidden,
        n_layers = args.n_layers,
    )
    
    paths = InputPaths(
        model_dir = args.model_dir,
        output_dir = args.output_data_dir,
        train_dir = args.train_dir,
        checkpoint_dir = args.checkpoint_dir or args.output_data_dir
    )

    os.makedirs(paths.model_dir, exist_ok=True)
    os.makedirs(paths.output_dir, exist_ok=True)

    print("Starting training")
    start_training(
        args.device,
        params,
        paths,
    )

if __name__ == "__main__":
    print("Running main")
    main()
