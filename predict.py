#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from train import load_checkpoint
from baseline_preprocess_input import one_hot_encode

def predict(net, char, device, use_embeddings, h=None, top_k=None):
    # lookup in a dict
    x = np.array([[net.char2int[char]]])
    if use_embeddings:
        inputs = torch.from_numpy(x)
    else:
        x = one_hot_encode(x, len(net.chars))
        inputs = torch.from_numpy(x)

    inputs = inputs.to(device)

    h = tuple([each.data for each in h])
    out, h = net(inputs, h)

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

def show_sample(net, size, device, use_embeddings, prime='The', top_k=None):
    net.to(device)

    net.eval() # eval mode
    chars = [ch for ch in prime]
    h = net.init_hidden(1)

    # this loop is for building up the hidden state for model's predictions
    # without it predictions would be very random
    for ch in prime:
        char, h = predict(net, ch, device, use_embeddings, h, top_k=top_k)

    # this is outside of the loop on purpose
    # we need only the last char
    chars.append(char)

    #  get a new char
    for ii in range(size):
        char, h = predict(net, chars[-1], device, use_embeddings, h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

def trim_spaces(text):
    '''
        Removes two and more spaces in a row.
    '''
    return " ".join(text.split())

def predict_by_word(text, input_words, n_words):
    '''
        Limits an output by number of the words requested.
    '''
    input_words = input_words.split(' ') # if more then one word in the input
    text_l = text.split(" ")
    
    output = text_l[:n_words + len(input_words)] # input shouldn't be counted for the len of prediction
    words = " ".join(output)
    return words

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicts next words after fisrt given")
    parser.add_argument("model_file", type=str, help="The name of the file with pretrained and saved model.")
    parser.add_argument("--n_words", type=int, help="number of words", required=True)
    parser.add_argument("--first-word", type=str, help="first word, start of prediction sequence", required=True)
    parser.add_argument("--device", type=str, help="device to use. default: cuda:0", default="cuda:0")

    args = parser.parse_args()

    model, _, _, _, _, _ = load_checkpoint(args.model_file, args.device)

    predicted = show_sample(model, 1000, args.device, use_embeddings=model.use_embeddings, prime=args.first_word, top_k=5)
    
    out = trim_spaces(predicted)
    result = predict_by_word(out, args.first_word, args.n_words)

    print(result)
# Example:
# ./predict.py models/{your model name}.pt --len 100 --first-word ja
