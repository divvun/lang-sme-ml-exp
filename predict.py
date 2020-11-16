#!/usr/bin/env python3

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from train import load_checkpoint
from encode_input import load_corpus_chars
from models import init_model
from baseline_preprocess_input import one_hot_encode

def predict(net, char, device,  use_embeddings=True, h=None, top_k=None):
    x = np.array([[net.char2int[char]]])

    if use_embeddings:
        inputs = torch.from_numpy(x)
    else:
    # tensor inputs
        x = one_hot_encode(x, len(net.chars))

        inputs = torch.from_numpy(x)

    inputs = inputs.to(device)
    h = tuple([each.data for each in h])

    out, h = net(inputs)

    # character probabilities from softmax
    p = F.softmax(out, dim=1).data
    p = p.cpu() # move to cpu

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    # returns encoded predicted value, and hidden_state
    return net.int2char[char], h

def show_sample(net, size, device, use_embeddings=True, prime='The', top_k=None):
    net.to(device)

    net.eval() # eval mode
    chars = [ch for ch in prime]
    h = net.init_hidden(1)

    for ch in prime:
        char, h = predict(net, ch, device, use_embeddings, h, top_k=top_k)

    chars.append(char)

    #  get a new char
    for ii in range(size):
        char, h = predict(net, chars[-1], device, use_embeddings, h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicts next words after fisrt given")
    parser.add_argument("model_file", type=str, help="The name of the file with pretrained and saved model.")
    parser.add_argument("--len", type=int, help="length of the sequence to be predicted", required=True)
    parser.add_argument("--first-word", type=str, help="first word, start of prediction sequence", required=True)
    parser.add_argument("--device", type=str, help="device to use. default: cuda:0", default="cuda:0")

    args = parser.parse_args()

    chars = load_corpus_chars()
    model_state_dict, opt_state_dict, tokens, n_hidden, n_layers, starting_epoch, is_gru, bidirectional, use_embeddings, emb_dim = load_checkpoint(args.model_file)
    model, opt = init_model(chars, args.device, is_gru, bidirectional, use_embeddings, emb_dim, n_hidden, n_layers)

    print(show_sample(model, args.len, args.device, prime=args.first_word, top_k=5))

# Example:
# ./predict.py models/{your model name}.pt --len 100 --first-word ja
