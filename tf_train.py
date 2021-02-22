import os
import argparse
import typing
import csv
import sys
import signal
from functools import lru_cache

# import torch
# import torch.nn as nn
import numpy as np 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical



# def get_data():
class Hyperparams(typing.NamedTuple):
    epochs: int
    batch_size: int
    lr: int = 0.0001
    bidirectional: bool = False
    emb_dim: int = 128
    is_gru: bool = False
    n_hidden: int = 756
    n_layers: int = 2

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
    def int2word_path(self):
        return os.path.join(self.train_dir, 'word2int.txt')

    @lru_cache(maxsize=None)
    def load_int2word(self):
        with open(self.int2word_path, 'r', encoding='utf-8') as f:
            return f.read()

    @property
    def int2pos_path(self):
        return os.path.join(self.train_dir, 'pos2int.txt')

    @lru_cache(maxsize=None)
    def load_int2pos(self):
        with open(self.int2pos_path, 'r', encoding='utf-8') as f:
            return f.read()

    @property
    def train_words_path(self):
        return os.path.join(self.train_dir, "all_words_enc.csv")

    @lru_cache(maxsize=None)
    def train_words(self):
        with open(self.train_words_path, 'r') as f:
            train = list(csv.reader(f))
          
            x_train = [int(i[0]) for i in train] 
            y_train = [int(i[2]) for i in train]

        return x_train, y_train
       
    # @property
    # def val_words_path(self):
    #     return os.path.join(self.train_dir, "val_words_enc.csv")

    # @lru_cache(maxsize=None)
    # def val_words(self):
    #     with open(self.val_words_path, 'r') as f:
    #         train = list(csv.reader(f))
           
    #         x_train = [int(i[0]) for i in train]
    #         pos_x_train = [int(i[1]) for i in train]
    #         y_train = [int(i[2]) for i in train]
    #         pos_y_train = [int(i[3]) for i in train]

    #     return (x_train, pos_x_train, y_train, pos_y_train)


def prep_input(paths: InputPaths, params : Hyperparams):

    vocab_size = len(paths.load_corpus_words())
    
    X,  y = paths.train_words()
    print("Loaded data! ", 'Tokens in vocabulary: ', vocab_size)
   
    seq_len = 50
    step = 10
    inputs = []
    targets = []

    input_X = []
    target_y = []
    word2int = paths.load_int2word()
    # split in sequences & prepare for 3d shape
    for i in range(0, len(X) - seq_len, step):
        inputs.append(X[i: i + seq_len])
        targets.append(y[i: i + seq_len])

        # input_X.append([word2int.get(word, 0) for word in inputs])
        # target_y.append(word2int.get(next_word, 0))

    print(f'Training seqs {len(inputs)}')
    # targets = to_categorical(targets, num_classes=vocab_size)
# 
    # 3d shape
    
    print('Reshaping...')
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)

    # word2int = paths.load_int2word()

    # inputs = np.zeros((len(inputs), seq_len, vocab_size), dtype=np.bool)
    # targets = np.zeros((len(targets), vocab_size), dtype=np.bool)
    
    print(inputs.shape)
    
    print(targets.shape)
    
    print('Rewriting reshaped training data...')
    with open('inputs.bin', 'wb') as f:
        np.save(f, inputs)
    with open('targets.bin', 'wb') as f:
        np.save(f, targets)

    # inputs = tf.keras.Input(shape=(None,), dtype="int64")
    

def run_training_loop(paths: InputPaths, params: Hyperparams):

    data = prep_input(paths,params)
    vocab_size = len(paths.load_corpus_words())
    inputs = np.load('inputs.bin')
    targets = np.load('targets.bin')
    # inputs = np.reshape(inputs.shape[0], inputs.shape[1], -1)
    # targets = np.squeeze(targets, axis=0)
    # targets = to_categorical(targets, num_classes=vocab_size)

    print(inputs.shape)
    print(targets.shape)
    seq_len = 50

    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_len))
    model.add(LSTM(256,input_shape=(seq_len, 2), return_sequences=True))
    # model.add(LSTM(100))
    # model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))

    # opt = tf.keras.optimizer.Adam(lr=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')


    print("Model's architecture: \n", model.summary())
    print(f"Training with batch size: {params.batch_size}")


    history = model.fit(inputs, targets, validation_split=0.2, batch_size=params.batch_size, epochs=1, shuffle=False).history
    model_n = 'lstm_test.h5'
    model.save(model_n)
     
    
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


print(run_training_loop(paths, params))