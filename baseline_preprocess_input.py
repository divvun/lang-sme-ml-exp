import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import re 
import csv
import subprocess
import linecache
from encode_words import load_whole_corpus

def one_hot_encode(arr, n_labels):
    
    # empty/only zeros array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    
    # put 1 where needed 
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot

class LazyTextDataset(Dataset):
    def __init__(self, filename):
        self._filename = filename
        self._total_data = 0
        self._total_data = int(subprocess.check_output("wc -l " + filename, shell=True).split()[0])
        # self._total_data = 0
        # with open(filename, "r") as f:
        #     self._total_data = len(f.readlines()) - 1

    def __getitem__(self, slice: slice):
        return_val = []
        start = slice.start if slice.start is not None else 0
        stop = slice.stop if slice.stop is not None else len(self)-1
        for idx in range(start, stop):
            return_val.append(self.read_line(idx + 1))
        return np.asarray(return_val)

    def read_line(self, line_num):
        line = linecache.getline(self._filename, line_num)
        csv_line = csv.reader([line])
        result = []
        for i in next(csv_line):
            result.append(int(i))
        return result
        # return int(" ".join(next(csv_line)))
      
    def __len__(self):
        return self._total_data

def get_batches(arr, batch_size, seq_length):
    # returns mini-batches of size batch_size*seq_lenth
    
    # arr = LazyTextDataset(arr)
    batch_size_total = batch_size * seq_length
    
    n_batches = len(arr)//batch_size_total
    # print(len(arr.dataset))
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

class Data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__(self):
        return len(self.X_data)

def get_batch_data(X, x_pos, y, y_pos, batch_size):
    X = np.asarray(X)
    x_pos = np.asarray(x_pos)
    y = np.asarray(y)
    y_pos = np.asarray(y_pos)

    X = X.reshape((-1, 1))
    x_pos = x_pos.reshape((-1,1))
    y = y.reshape((-1, 1))
    y_pos = y_pos.reshape((-1, 1))

    X = np.concatenate((X, x_pos), axis=1)
    y = np.concatenate((y, y_pos), axis=1)
   
    data = Data(X, y)

    batched_data = DataLoader(dataset=data, num_workers=4, batch_size=batch_size, drop_last=True, shuffle=False)
    # for x, y in batched_data:
    #     print(x)
    #     print(y)
    #     exit()
    return batched_data


def get_batch(path, b_size):
    # path to 'train_words_enc.csv' or 'val_words_enc.csv'
    
    X, x_pos, y, y_pos = load_whole_corpus(path)

    data = Data(X, y)
    data_pos = Data(x_pos, y_pos)

    batched = DataLoader(dataset=data, batch_size=b_size, drop_last=True, shuffle=False)

    batched_pos = DataLoader(dataset=data_pos, batch_size=b_size, drop_last=True, shuffle=False)

    return (batched, batched_pos)


