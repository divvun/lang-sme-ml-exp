#!/usr/bin/env python3

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import argparse
import torch
import torch.nn as nn
import numpy as np
from baseline_preprocess_input import get_batches, one_hot_encode
from encode_input import load_corpus_chars, load_encoded_corpus
from models import init_model
import json

def save_checkpoint(model, opt, tokens, n_hidden, n_layers, epoch, path):
    torch.save({
        'n_hidden': model.n_hidden,
        'n_layers': model.n_layers,
        'model_state_dict': model.state_dict(),
        'opt': opt.state_dict(),
        'tokens': model.chars,
        'epoch':epoch,

    }, path)

def load_checkpoint(path, model, opt):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['opt'])
    tokens = checkpoint['tokens']
    n_hidden = checkpoint['n_hidden']
    n_layers = checkpoint['n_layers']
    epoch = checkpoint['epoch']

    return model, opt, tokens, n_hidden, n_layers, epoch


def train(device, model_name='sme_rnn', epochs=10, batch_size=10, seq_length=50, lr=0.001, resume_from_saved=True, bidirectional=False, use_embeddings=False, emb_dim=128, is_gru=False, n_hidden=756, n_layers=2):
    current_params = {
        'device': device,
        'is_gru': is_gru,
        'bidirectional': bidirectional,
        'use_embeddings':use_embeddings,
        'emb_dim' : emb_dim,
        'n_hidden' : n_hidden,
        'n_layers': n_layers,
        'lr': lr
    }
    with open(f"./paramaters/{model_name}_params.txt", "w") as f:
        f.write(json.dumps(current_params))

    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_PATH = f'models/{model_name}.pt'
    clip=5
    val_frac=0.1

    chars = load_corpus_chars()

    if resume_from_saved:
        model, opt = init_model(chars, device, is_gru, bidirectional, use_embeddings, emb_dim, n_hidden, n_layers, lr)

        model, opt, tokens, n_hidden, n_layers, starting_epoch = load_checkpoint(CHECKPOINT_PATH, model, opt)
        # print("LOADED", n_hidden, n_layers, starting_epoch, is_gru, bidirectional, use_embeddings, emb_dim)
        starting_epoch += 1
        print(model)
        print(f"Resuming {model_name} from epoch {starting_epoch+1} ...")
        model.to(device)
    else:
        #initialize model (do it in the if in case somebody mismatched the parameters and selected resume from saved)
        model, opt = init_model(chars, device, is_gru, bidirectional, use_embeddings, emb_dim, n_hidden, n_layers, lr)
        print("Model's architecture: \n", model)
        starting_epoch = 0
        print(f"Starting to train {model_name}...")

    #train mode

    model.train()

    data = load_encoded_corpus()
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    n_hidden = model.n_hidden
    n_layers  = model.n_layers
    model.to(device)

    tokens = len(model.chars)
    for e in range(starting_epoch, epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)
        # h = h.to(device)

        for x, y in get_batches(data, batch_size, seq_length):

            if use_embeddings:
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                inputs, targets = inputs.to(device), targets.to(device)

            else:
                # One-hot encode, make tensor move to cuda
                x = one_hot_encode(x, tokens)
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                inputs, targets = inputs.to(device), targets.to(device)

            # Creating a separate variables for the hidden state
            h = tuple([each.data for each in h])
            # h = h.to(device)
            model.zero_grad()
            # print(inputs.shape)
            output, h = model(inputs, h)
            # some reshaping for output needed
            if bidirectional:
                output = output.view(output.size(0), output.size(2), output.size(1))
                loss = criterion(output, targets)
            else:
                loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            if resume_from_saved:
                model.cpu()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            # put back to cuda if was detached
            model.to(device)

        save_checkpoint(model, opt, tokens, n_hidden, n_layers, e, CHECKPOINT_PATH)
        # return current_params
        print("Epoch: {}...".format(e+1),
            "Loss: {:.4f}...".format(loss.item()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="paramaters for training the model")
    parser.add_argument("--device", type=str, help="device to train on: cuda or cpu", required=True)
    parser.add_argument("--model-name", type=str, help="name of model to be saved", required=True)
    parser.add_argument("--epochs", type=int, help="num of epoch to train over", required=True)
    parser.add_argument("--batch-size", type=int, help="mini_batch size", required=True)
    parser.add_argument("--seq-len", type=int, help="max length of a sequence to be trained on", required=True)
    parser.add_argument("--lr", type=float, help="learning rate: default 0.0001", default=0.0001)
    parser.add_argument("--resume-from-saved", help="train from scratch or contionue training from saved", action='store_true')
    parser.add_argument("--bidirectional", help="whether a model is bidirectonal: default False", action='store_true')
    parser.add_argument("--use-emb", help="whether to ude embedding layer: default True", action='store_true')
    parser.add_argument("--emb-dim", type=int, help="size of emb_dim", default=128)
    parser.add_argument("--is-gru", help="whether to use GRU or LSTM model", action='store_true')
    parser.add_argument("--n-hidden", type=int, help="size of hidden layer", default=756)
    parser.add_argument("--n-layers", type=int, help="num of layers in the model", default=2)

    args = parser.parse_args()

    train(
        device=args.device,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_len,
        lr=args.lr,
        resume_from_saved=args.resume_from_saved,
        bidirectional=args.bidirectional,
        use_embeddings=args.use_emb,
        emb_dim=args.emb_dim,
        is_gru=args.is_gru,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
    )

# Example:
# ./train.py --device=cuda:0 --model-name {your model name} --epochs 1 --batch-size 128 --seq-len 300
