import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import re 

with open('data/sme-freecorpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# clean very special char
text = text.replace("¶", "").replace('•', '').replace('□', '').replace('§', '').replace('\uf03d', '').replace('π', '').replace('●', '').replace('µ', '').replace('º', '').replace('文', '').replace('中', '').replace('⅞', '').replace('½', '').replace('⅓', '').replace('¾', '').replace('¹', '').replace('³', '').replace('\t', '')
# remove numbers
text = re.sub(r'[0-9]+', '', text)
# remove russian text (it is in data)
text = re.sub(r"[А-Яа-я]", '', text) 
# remove puctuation
text = re.sub(r"[^\w\s]", "", text) 

# encode the text 
# 1. int2char, integers to characters
# 2. char2int, characters to unique integers
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# encode the text
encoded = np.array([char2int[ch] for ch in text])

def one_hot_encode(arr, n_labels):
    
    # eempty/only zeros array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    
    # put 1 where needed 
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot

def get_batches(arr, batch_size, seq_length):
    # returns mini-batches of size batch_size*seq_lenth
    
    batch_size_total = batch_size * seq_length
    
    n_batches = len(arr)//batch_size_total
    
    arr = arr[:n_batches * batch_size_total]
    arr = arr.reshape((batch_size, -1))
 
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one (because the next char or word)
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

