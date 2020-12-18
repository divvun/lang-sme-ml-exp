#!/usr/bin/env python3
import subprocess
import argparse
import torch
import torch.nn as nn
import numpy as np 
from baseline_preprocess_input import get_batches, one_hot_encode, LazyTextDataset, get_batch_data
from encode_input import load_corpus_chars, load_encoded_corpus
from encode_words import load_corpus_words, load_whole_corpus
from models import init_model, init_opt
# from words_batching import batch_large_data

def save_checkpoint(model, opt, epoch, lr, batch_size, path):
    torch.save({
        'model': model.state_dict(),
        'is_gru': model.is_gru,
        'bidirectional': model.bidirectional,
        'emb_dim': model.emb_dim,
        'n_hidden': model.n_hidden,
        'n_layers': model.n_layers,
        'drop_prob': model.drop_prob,
        'lr' : lr,
        'opt': opt.state_dict(),
        'epoch': epoch,
        'batch_size': batch_size,
    }, path)

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=torch.device(device))
    tokens = load_corpus_words()
    model = init_model(tokens, device, checkpoint['is_gru'], checkpoint['bidirectional'], checkpoint['emb_dim'], checkpoint['n_hidden'], checkpoint['n_layers'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    opt = init_opt(model, checkpoint['lr'])
    opt.load_state_dict(checkpoint['opt'])

    return model, opt, checkpoint['epoch'], checkpoint['lr'], checkpoint['batch_size']


def checkpoint_path(model_name):
    return f'models/{model_name}.pt'

def resume_training(device, model_name='sme_rnn', epochs=10): 
    model, opt, starting_epoch, lr, batch_size = load_checkpoint(checkpoint_path(model_name), device)

    starting_epoch += 1

    print(f"\nResuming {model_name} from epoch {starting_epoch+1} ...")

    run_training_loop(model_name, model, opt, device, starting_epoch, epochs, lr, batch_size)

def start_training(device, model_name='sme_rnn', epochs=10, batch_size=10, lr=0.0001, bidirectional=False, emb_dim=128, is_gru=False, n_hidden=756, n_layers=2):
    
    tokens = load_corpus_words()
  
    model = init_model(tokens, device, is_gru, bidirectional, emb_dim, n_hidden, n_layers)
    opt = init_opt(model, lr)
    starting_epoch = 0

    print(f"\nStarting to train {model_name}...")
    run_training_loop(model_name, model, opt, device, starting_epoch, epochs, lr, batch_size)

def run_training_loop(model_name, model, opt, device, starting_epoch, epochs, lr, batch_size):

    print("Model's architecture: \n", model)
    print(f"Training with batch size: {batch_size}")

    clip=5

    criterion = nn.CrossEntropyLoss()

    #train mode
    model.to(device) 
    model.train()
    
    
    x_train, x_pos_train, y_train, y_pos_train, x_val, x_pos_val, y_val, y_pos_val = load_whole_corpus()
       

    tokens = len(model.tokens)
    for e in range(starting_epoch, epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)
      
        for x, y in get_batch_data(x_train, x_pos_train, y_train, y_pos_train, batch_size):
          
            inputs, targets = x.to(device), y.to(device)
          
            # Creating a separate variables for the hidden state
            if not model.is_gru:
                h = tuple([each.data for each in h])
        
            model.zero_grad()
           
            output, h = model(inputs, h)
            # some reshaping for output needed
            output = output.view(output.size(0), output.size(2), output.size(1))
            loss = criterion(output, targets)
                
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        
        val_h = model.init_hidden(batch_size)
        val_losses = []

        # move to eval mode
        model.eval()

        for x, y in get_batch_data(x_val, x_pos_val, y_val,y_pos_val, batch_size):
         
            inputs, targets = x.to(device), y.to(device)
           
            if not model.is_gru:
                val_h = tuple([each.data for each in val_h])
            
            output, val_h = model(inputs, val_h)
            
            output = output.view(output.size(0), output.size(2), output.size(1))
            val_loss = criterion(output, targets)
           
            val_losses.append(val_loss.item())
        
        model.train() # reset to train 

        save_checkpoint(model, opt, e, lr, batch_size, checkpoint_path(model_name))
        # return current_params
        print("Epoch: {}...".format(e+1),
            "Loss: {:.4f}...".format(loss.item()),
            "Valid loss: {:.4f}".format(np.mean(val_losses)))

def add_required_args(p):
    p.add_argument("--device", type=str, help="cuda or cpu", required=True)
    p.add_argument("--model-name", type=str, help='name of model to be saved', required=True)
    p.add_argument("--epochs", type=int, help='num epochs to train over', required=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="paramaters for training the model")

    subparsers = parser.add_subparsers()

    resume_saved_parser = subparsers.add_parser(name='resume', help='continue training a saved model')

    resume_saved_parser.set_defaults(which='resume')

    add_required_args(resume_saved_parser)
    
    
    start_parser = subparsers.add_parser(name='start', help='start training from scratch')
    start_parser.set_defaults(which='start')

    add_required_args(start_parser)

    start_parser.add_argument("--lr", type=float, help="learning rate: default 0.0001", default=0.0001)
    start_parser.add_argument("--batch-size", type=int, help="mini_batch size", required=True)
    start_parser.add_argument("--bidirectional", help="whether a model is bidirectonal: default False", action='store_true')
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
            args.lr,
            args.bidirectional,
            args.emb_dim,
            args.is_gru,
            args.n_hidden,
            args.n_layers,
        )
 
# Example:
# ./train.py start --device=cuda:0 --model-name {your model name} --epochs 1 --batch-size 128 
# ./train.py resume --device=cuda:0 --model-name {your model name} --epochs 5
