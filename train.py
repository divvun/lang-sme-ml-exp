#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
from baseline_preprocess_input import get_batches, one_hot_encode
from encode_input import load_corpus_chars, load_encoded_corpus
from models import init_model, init_opt

def save_checkpoint(model, opt, epoch, lr, batch_size, seq_length, path):
    torch.save({
        'model': model,
        'opt': opt.state_dict(),
        'lr': lr,
        'epoch': epoch,
        'batch_size': batch_size,
        'seq_length': seq_length,
    }, path)

def load_checkpoint(path, device):
    checkpoint = torch.load(path)
    checkpoint['model'].device = device
    checkpoint['model'].to(device)
    opt = init_opt(checkpoint['model'], checkpoint['lr'])
    opt.load_state_dict(checkpoint['opt'])
    return checkpoint['model'], opt, checkpoint['epoch'], checkpoint['lr'], checkpoint['batch_size'], checkpoint['seq_length']

def checkpoint_path(model_name):
    return f'models/{model_name}.pt'

def resume_training(device, model_name='sme_rnn', epochs=10):
    model, opt, starting_epoch, lr, batch_size, seq_length = load_checkpoint(checkpoint_path(model_name), device)
    starting_epoch += 1

    print(f"Resuming {model_name} from epoch {starting_epoch+1} ...")
    run_training_loop(model_name, model, opt, device, starting_epoch, epochs, lr, batch_size, seq_length)

def start_training(device, model_name='sme_rnn', epochs=10, batch_size=10, seq_length=50, lr=0.001, bidirectional=False, use_embeddings=False, emb_dim=128, is_gru=False, n_hidden=756, n_layers=2):
    if not use_embeddings:
        emb_dim = None

    chars = load_corpus_chars()
    model = init_model(chars, device, is_gru, bidirectional, use_embeddings, emb_dim, n_hidden, n_layers)
    opt = init_opt(model, lr)
    starting_epoch = 0

    print(f"Starting to train {model_name}...")
    run_training_loop(model_name, model, opt, device, starting_epoch, epochs, lr, batch_size, seq_length)

def run_training_loop(model_name, model, opt, device, starting_epoch, epochs, lr, batch_size, seq_length):
    print("Model's architecture: \n", model)
    print(f"Batch size: {batch_size}; Seq length: {seq_length}")

    clip=5
    val_frac=0.1

    criterion = nn.CrossEntropyLoss()

    #train mode
    model.to(device)
    model.train()

    data = load_encoded_corpus()
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    tokens = len(model.chars)
    for e in range(starting_epoch, epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)
        # h = h.to(device)

        for x, y in get_batches(data, batch_size, seq_length):
            if model.use_embeddings:
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                inputs, targets = inputs.to(device), targets.to(device)
            else:
                # One-hot encode, make tensor move to cuda
                x = one_hot_encode(x, tokens)
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                inputs, targets = inputs.to(device), targets.to(device)

            # Creating a separate variables for the hidden state
            if not model.is_gru:
                h = tuple([each.data for each in h])
            # h = h.to(device)
            model.zero_grad()
            # print(inputs.shape)
            output, h = model(inputs, h)
            # some reshaping for output needed
            if model.bidirectional:
                output = output.view(output.size(0), output.size(2), output.size(1))
                loss = criterion(output, targets)
            else:
                loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            # put back to cuda if was detached
            model.to(device)

        save_checkpoint(model, opt, e, lr, batch_size, seq_length, checkpoint_path(model_name))
        # return current_params
        print("Epoch: {}...".format(e+1),
            "Loss: {:.4f}...".format(loss.item()))

def add_required_args(p):
    p.add_argument("--device", type=str, help="device to train on: cuda or cpu", required=True)
    p.add_argument("--model-name", type=str, help="name of model to be saved", required=True)
    p.add_argument("--epochs", type=int, help="num of epoch to train over", required=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="paramaters for training the model")

    subparsers = parser.add_subparsers()

    resume_saved_parser = subparsers.add_parser(name="resume", help="continue training a saved model")
    resume_saved_parser.set_defaults(which="resume")
    add_required_args(resume_saved_parser)

    start_parser = subparsers.add_parser(name="start", help="start training a model from scratch")
    start_parser.set_defaults(which="start")
    add_required_args(start_parser)
    start_parser.add_argument("--lr", type=float, help="learning rate: default 0.0001", default=0.0001)
    start_parser.add_argument("--batch-size", type=int, help="mini_batch size", required=True)
    start_parser.add_argument("--seq-len", type=int, help="max length of a sequence to be trained on", required=True)
    start_parser.add_argument("--bidirectional", help="whether a model is bidirectonal: default False", action='store_true')
    start_parser.add_argument("--use-emb", help="whether to ude embedding layer: default True", action='store_true')
    start_parser.add_argument("--emb-dim", type=int, help="size of emb_dim", default=128)
    start_parser.add_argument("--is-gru", help="whether to use GRU or LSTM model", action='store_true')
    start_parser.add_argument("--n-hidden", type=int, help="size of hidden layer", default=756)
    start_parser.add_argument("--n-layers", type=int, help="num of layers in the model", default=2)

    args = parser.parse_args()

    if args.which == 'resume':
        resume_training(args.device, args.model_name, args.epochs)
    else:
        start_training(
            args.device,
            args.model_name,
            args.epochs,
            args.batch_size,
            args.seq_len,
            args.lr,
            args.bidirectional,
            args.use_emb,
            args.emb_dim,
            args.is_gru,
            args.n_hidden,
            args.n_layers
        )

# Example:
# ./train.py start --device=cuda:0 --model-name {your model name} --epochs 1 --batch-size 128 --seq-len 100
# ./train.py resume --device=cuda:0 --model-name {your model name} --epochs 5
