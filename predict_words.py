#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from train import load_checkpoint
from baseline_preprocess_input import one_hot_encode
from encode_words import load_corpus_words

def predict_by_word(kept_word, input_word, n_preds, net, device, use_embeddings, h=None, top_k=None):
    
    predicted = []
    for i in range(n_preds):

        pred_word, hs = predict(net, input_word, device, use_embeddings, h, top_k=top_k)
    
        if pred_word not in predicted:
            predicted += [' '.join(kept_word) + ' ' + str(input_word) + ' ' + str(pred_word)]
    return predicted

def complete_word(prime, net, n_preds):

    tokens = load_corpus_words()
    possible_suggestions = [t for t in tokens if t.startswith(prime)]
    # chooses top n frequent words
    frequecy_suggestions = sorted([net.item2int[s] for s in possible_suggestions])[:n_preds]

    suggestions = [net.int2item[s] for s in frequecy_suggestions]

    return suggestions

def predict(net, word, device, use_embeddings, h=None, top_k=None):
    # lookup in a dict

    x = np.array([[net.item2int[word]]])
    if use_embeddings:
        inputs = torch.from_numpy(x)
    else:
        x = one_hot_encode(x, len(net.tokens))
        inputs = torch.from_numpy(x)

    inputs = inputs.to(device)

    h = tuple([each.data for each in h])
    out, h = net(inputs, h)

    # word probabilities from softmax
    p = F.softmax(out, dim=1).data
    p = p.cpu() # move to cpu

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(net.tokens))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    # returns encoded predicted value, and hidden_state
    return net.int2item[char], h

def show_sample(net, n_preds, device, use_embeddings, prime='The', top_k=None):
    net.to(device)

    net.eval() # eval mode
    # chars = [ch for ch in prime]
    predictions = []
    h = net.init_hidden(1)

    # this loop is for building up the hidden state for model's predictions
    # without it predictions would be very random
    # if len(prime.split(' ')) > 1:
    #     w = prime.split(' ')[0]
    #     # for w in prime.split(' '):
    #     word, h = predict(net, w, device, use_embeddings, h, top_k=top_k)
    # else:
    #     #TODO finish
    #     word, h = predict(net, prime, device, use_embeddings, h, top_k=top_k)

    # this is outside of the loop on purpose
    # we need only the last word
    # words.append(word)
    # print(words)
    #  get a new char


    # check if the input is finihsed with a white space
    if prime.endswith(' '):
        print('\n')
        print('Got completed word(s) as input. Predicting next words ...\n')
        # keep words preceding the last input words
        # this also will be further used when model uses pos tags
        # for now it's just kept for consistent print stats
        if len(prime.split()) > 1:
            kept_input = prime.split()[:-1]

            while len(predictions) < n_preds:

                input_word = prime.split()[-1]

                word, hs = predict(net, input_word, device, use_embeddings, h, top_k=top_k)
                prediction = ' '.join(kept_input) + " " + input_word + " " + word

                if prediction not in predictions:
                    predictions.append(''.join(prediction))
        print(predictions)
    if not prime.endswith(' '):
        print('\n')
        print('Got an unfinished word! Autocompleting it and predicting the next one ...\n')
        if len(prime.split()) > 1:
            kept_input = prime.split()[:-1]
            
            input_words = complete_word(prime.split()[-1], net, n_preds)
            for input_word in input_words:

                res = predict_by_word(kept_input, input_word, n_preds, net, device, use_embeddings, h, top_k=top_k)
                predictions += res
        else:
            input_words = complete_word(prime.split()[-1], net, n_preds)
            # print(input_words)
            for input_word in input_words:
                print('here')
                res = predict_by_word(kept_input, input_word, n_preds, net, device, use_embeddings, h, top_k=top_k)
                predictions += res
                
    
    return '\n'.join(p for p in predictions)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicts next words after fisrt given")
    parser.add_argument("model_file", type=str, help="The name of the file with pretrained and saved model.")
    parser.add_argument("--n-preds", type=int, help="number of predictions after --first-word", required=True)
    parser.add_argument("--first-word", type=str, help="first word, start of prediction sequence", required=True)
    parser.add_argument("--device", type=str, help="device to use. default: cuda:0", default="cuda:0")
    args = parser.parse_args()

    model, _, _, _, _, _ = load_checkpoint(args.model_file, args.device)

    result = []
    # for i in range(args.n_preds):

    predicted = show_sample(model, args.n_preds, args.device, use_embeddings=model.use_embeddings, prime=args.first_word, top_k=5)

        # out = trim_spaces(predicted)
        # res = predict_by_word(predicted, args.first_word)

        # result.append(res)

    print(predicted)

# Example:
# ./predict.py models/{your model name}.pt --n-preds 3 --first-word ja
