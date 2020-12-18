#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from train import load_checkpoint
# from baseline_preprocess_input import one_hot_encode
from encode_words import load_corpus_words
import json
import subprocess

def get_most_common(word2int_path, int2item_path, n_preds):

    with open(word2int_path) as f:
        word2int = json.load(f)

    with open(int2item_path) as f:
        int2word = json.load(f)

    frequecy_suggestions = sorted(word2int[w] for w in word2int)[:n_preds]
    
    suggestions = [int2word[str(s)] for s in frequecy_suggestions]

    return suggestions

def check_word(prime):

    tokens = load_corpus_words()

    for k in tokens:
        if k.startswith(prime):
            return True
        
    return False

def predict_by_word(word2int_path, int2word_path, input_word, n_preds, net, device, h=None, top_k=None):

    predicted = []
    while len(predicted) < n_preds:

        pred_word, hs = predict(net, word2int_path, int2word_path, input_word, device, h, top_k=top_k)
    
        predicted += [' '.join(input_word) + ' ' + str(pred_word)]
        
    return predicted

def complete_word(prime, word2int_path, int2item_path, n_preds):

    tokens = load_corpus_words()
    possible_suggestions = [t for t in tokens if t.startswith(prime)]
    # chooses top n frequent words from word2int.txt
    with open(word2int_path, 'r') as f:
        item2int = json.load(f)

    frequecy_suggestions = sorted([item2int[s] for s in possible_suggestions])[:n_preds]
    with open(int2item_path, 'r') as f:
        int2item = json.load(f)

    suggestions = [int2item[str(s)] for s in frequecy_suggestions]
    return suggestions

def predict(net, word2int_path, int2word_path, word, device, h=None, top_k=None):
    # lookup in a dict
    with open(word2int_path, 'r') as f:
        item2int = json.load(f)

    with open(int2word_path, 'r') as f:
        int2item = json.load(f)

    if len(word) > 1:
        x = []
        for w in word:
            w = item2int[w]
            x.append(w)          
    else:
        word = ''.join(word)
        x = [int(item2int[word])]
        word = word.split()

    pos_ouput = []
    # print('getting pos tag!')
    for w in word:
        pos_ouput.append(subprocess.check_output(f'echo {w}|hfst-tokenise ~/usr/share/giella/sme/tokeniser-disamb-gt-desc.pmhfst |hfst-lookup -q ~/usr/share/giella/sme/analyser-gt-norm.hfstol', shell=True).decode('utf-8'))
    
    pos = []
    for out in pos_ouput:
        pos.append(out.split('\t')[1].split('+')[1])  # if pos in valid_pos list

    with open('./pos2int.txt', 'r') as f:
        pos2int = json.load(f)
    with open('./int2pos.txt', 'r') as f:
        int2pos = json.load(f)
    
    pos_enc = [pos2int[p] for p in pos]
   
    inputs = np.array([[a,b] for a, b in zip(x, pos_enc)]) # [[word, pos] [word pos]]
    
    inputs = torch.from_numpy(inputs)
   
    inputs = inputs.to(device)
    
    h = tuple([each.data for each in h])
    # print('inputs', inputs)
    out, h = net(inputs, h)
    # word probabilities from softmax
    p = F.softmax(out, dim=1).data
    p = p.cpu() # move to cpu
    # print(p[2])
    if top_k is None:
        top_ch = np.arange(len(tokens))
    else:
        p, top_ch = p[len(word)-1].topk(top_k)
        top_ch = top_ch.numpy().squeeze()
    
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch[0], p=p[0]/p[0].sum())
   
    return int2item[str(char)], h

def show_sample(net, word2int_path, int2item_path, n_preds, device, prime='The', top_k=None):

    net.to(device)
    forget_point = 3
   
    net.eval() # eval mode
   
    predictions = []

    # check if the input is finihsed with a white space
    if prime.endswith(' '):
        print('\n')
        print("Got complete word(s). Predicting next words ... \n")
        prime = prime.split()
        is_in_dict = check_word(prime[-1])
        if not is_in_dict:
            print('Got nonsense word!')
            prediction = get_most_common(word2int_path, int2item_path, n_preds)
            predictions = [' '.join(prime) + p for p in prediction]
        else:
            # print('Will consider only real words in the input! \n')
            valid_prime = []
            for p in prime:
                if check_word(p):
                    valid_prime.append(p)
            prime = valid_prime
            h = net.init_hidden(len(prime))
            if len(prime) > forget_point:
                prime = prime[-forget_point:]
                h = net.init_hidden(forget_point)
                prime = prime[-forget_point:]
            while len(predictions) < n_preds:

                # input_word = prime.split()[-1]
                word, hs = predict(net, word2int_path, int2item_path, prime, device, h, top_k=top_k)
                prediction = ' '.join(prime) + " " + word

                if prediction not in predictions:
                    predictions.append(''.join(prediction))

    else:
        print('\n')
        print('Got an unfinished word! Autocompleting it and predicting the next one ...\n')
        # if len(prime.split(' ')) > 1:
        prime = prime.split()
        is_in_dict = check_word(prime[-1])
        if not is_in_dict:
            print('Got nonsense word! \n')
            prediction = get_most_common(word2int_path, int2item_path, n_preds)
            predictions = [' '.join(prime) + ' ' + p for p in prediction]  
        else:
            # print('Will consider only real words in the input! \n')
            valid_prime = []
            for p in prime:
                if check_word(p):
                    valid_prime.append(p)

            prime = valid_prime
            h = net.init_hidden(len(prime))

            if len(prime) > forget_point:
                h = net.init_hidden(forget_point)
                prime = prime[-forget_point:]
            
            input_words = complete_word(prime[-1], word2int_path, int2item_path, n_preds)
            for input_word in input_words:
                if len(prime) > 1:

                    inputs = ' '.join(prime[:-1]) + " " + input_word
                    inputs = inputs.split()

                else:
                    inputs = input_word.split()

                res = predict_by_word(word2int_path, int2item_path, inputs, n_preds, net, device, h, top_k=top_k)
                predictions += res
                predictions.append('')

    return '\n'.join(p for p in predictions)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicts next words after fisrt given")
    parser.add_argument("model_file", type=str, help="The name of the file with pretrained and saved model.")
    parser.add_argument("--n-preds", type=int, help="number of predictions after --first-word", required=True)
    parser.add_argument("--first-word", type=str, help="first word, start of prediction sequence", required=True)
    parser.add_argument("--device", type=str, help="device to use. default: cuda:0", default="cuda:0")
    args = parser.parse_args()

    model, _, _, _, _ = load_checkpoint(args.model_file, args.device)
    result = []
    predicted = show_sample(model, './word2int.txt', './int2word.txt', args.n_preds, args.device, prime=args.first_word, top_k=5)

    print(predicted)

# Example:
# ./predict.py models/{your model name}.pt --n-preds 3 --first-word 'ja'
