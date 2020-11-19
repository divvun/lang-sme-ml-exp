# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import re 

def one_hot_encode(arr, n_labels):
    
    # empty/only zeros array
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

