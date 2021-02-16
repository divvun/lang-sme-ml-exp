#!/usr/bin/env python3
import os
import subprocess
import argparse
import torch
import torch.nn as nn
import numpy as np 
from baseline_preprocess_input import get_batches, one_hot_encode, LazyTextDataset, get_batch
from encode_input import load_corpus_chars, load_encoded_corpus
from encode_words import load_corpus_words, load_whole_corpus, load_pos
from models import init_model, init_opt, RNN
# from words_batching import batch_large_data

def save_checkpoint(model, opt, epoch, lr, batch_size, model_name):
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
    }, model_name)

def load_checkpoint(model_name, device):
    checkpoint = torch.load(model_name, map_location=torch.device(device))
    tokens = load_corpus_words()
    pos = load_pos()
    model = init_model(tokens, pos, device, checkpoint['is_gru'], checkpoint['bidirectional'], checkpoint['emb_dim'], checkpoint['n_hidden'], checkpoint['n_layers'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    opt = init_opt(model, checkpoint['lr'])
    opt.load_state_dict(checkpoint['opt'])

    return model, opt, checkpoint['epoch'], checkpoint['lr'], checkpoint['batch_size']


def checkpoint_path(model_name):
    return f'models/{model_name}.pth'

def resume_training(device, model_name='sme_rnn', model_dir, epochs=10): 
    model, opt, starting_epoch, lr, batch_size = load_checkpoint(checkpoint_path(model_name), device)

    starting_epoch += 1

    print(f"\nResuming {model_name} from epoch {starting_epoch+1} ...")

    run_training_loop(model_name, model, opt, device, starting_epoch, epochs, lr, batch_size, train_dir, val_dir)

def start_training(device, train_dir, val_dir, model_name='sme_rnn', model_dir, epochs=10, batch_size=10, lr=0.0001, bidirectional=False, emb_dim=128, is_gru=False, n_hidden=756, n_layers=2):
    #
    tokens = load_corpus_words()
    pos = load_pos()
    model = init_model(tokens, pos, device, is_gru, bidirectional, emb_dim, n_hidden, n_layers)
    opt = init_opt(model, lr)
    starting_epoch = 0

    print(f"\nStarting to train {model_name}...")
    run_training_loop(model_name, model, opt, device, starting_epoch, epochs, lr, batch_size, train_dir, val_dir)

def run_training_loop(model_name, model, opt, device, starting_epoch, epochs, lr, batch_size, train_dir, val_dir):

    print("Model's architecture: \n", model)
    print(f"Training with batch size: {batch_size}")

    clip=5

    criterion = nn.CrossEntropyLoss()

    #train mode
    model.to(device) 
    model.train()
    
    for e in range(starting_epoch, epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)
      
        tok, pos = get_batch(train_dir, batch_size) # train-dir = 'train_words_enc.csv'

        for (tok_x, tok_y), (pos_x, pos_y) in zip(tok, pos):
            
          
            inputs_tok, targets_tok = tok_x.to(device), tok_y.to(device)
            inputs_pos, targets_pos = pos_x.to(device), pos_y.to(device)

        # Creating a separate variables for the hidden state
            if not model.is_gru:
                h = tuple([each.data for each in h])
    
            model.zero_grad()
        
            output, h = model(inputs_tok, inputs_pos, h)
            # some reshaping for output needed
         
            output = output.view(batch_size, -1)
            loss = criterion(output, targets_tok)
            # print(loss)  
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        
        val_h = model.init_hidden(batch_size, batch_size)
        val_losses = []

        # move to eval mode
        model.eval()

        tok, pos = get_batch(val_dir, batch_size)  # val-dir 'val_words_enc.csv'

        for (tok_x, tok_y), (pos_x, pos_y) in zip(tok, pos):
  
            inputs_tok, targets_tok = tok_x.to(device), tok_y.to(device)
            inputs_pos, targets_pos = pos_x.to(device), pos_y.to(device)

            if not model.is_gru:
                val_h = tuple([each.data for each in val_h])
            
            output, val_h = model(inputs_tok, inputs_pos, val_h)
            
            output = output.view(batch_size,-1)
            val_loss = criterion(output, targets_tok)
           
            val_losses.append(val_loss.item())
        
        model.train() # reset to train 

        save_checkpoint(model, opt, e, lr, batch_size, checkpoint_path(model_name))
        # return current_params
        print("Epoch: {}...".format(e+1),
            "Loss: {:.4f}...".format(loss.item()),
            "Valid loss: {:.4f}".format(np.mean(val_losses)))

# def model_fn(model_dir):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tokens = load_corpus_words()
#     pos = load_pos()
#     checkpoint = torch.load(model_dir, map_location=torch.device(device))

#     model = init_model(tokens, pos, device, checkpoint['is_gru'], checkpoint['bidirectional'], checkpoint['emb_dim'], checkpoint['n_hidden'], checkpoint['n_layers'])
    
#     model = torch.nn.DataParallel(model)

#     with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
#         model.load_state_dict(torch.load(f))
#         model.rnn.flatten_paramaters()

#     model.to(device)
#     return {'model':model, 'tokens':tokens}

def add_required_args(p):
    p.add_argument("--device", type=str, help="cuda or cpu", required=True)
    p.add_argument("--model-name", type=str, help='name of model to be saved', required=True)
    p.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
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

    # env args
    # start_parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    start_parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    start_parser.add_argument('--val-dir', type=str, default=os.environ['SM_CHANNEL_VAL'])
    # start_parser.add_argument('--vocab', type=str, default=os.environ["SM_CHANNEL_VAL"])
    args = parser.parse_args()

    if args.which == 'resume':
        resume_training(args.device, args.model_name, args.model_dir, args.epochs)

    else:
        start_training(
            args.device,
            args.model_name,
            args.train_dir, 
            args.val_dir, 
            args.model_dir,
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
