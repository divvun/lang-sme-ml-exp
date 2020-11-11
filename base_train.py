import torch
import torch.nn as nn
import numpy as np
from baseline_preprocess_input import get_batches, one_hot_encode


def save_checkpoint(model, opt, tokens, n_hidden, n_layers, epoch, path):

    torch.save({
        'n_hidden': model.n_hidden,
        'n_layers': model.n_layers,
        'model_state_dict': model.state_dict(),
        'opt': opt.state_dict(),
        'tokens': model.chars,
        'epoch':epoch,
    }, path)


def load_checkpoint(path, model, opt):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['opt'])
    tokens = checkpoint['tokens']
    n_hidden = checkpoint['n_hidden']
    n_layers = checkpoint['n_layers']
    
    return model, opt, tokens, n_hidden, n_layers, checkpoint['epoch']


def train_and_save(model, data, device, model_name='sme_rnn', epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, resume_from_saved=True, bidirectional=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
   
    criterion = nn.CrossEntropyLoss()
    
    CHECKPOINT_PATH = f'{model_name}'

    if resume_from_saved:
        
        model, opt, tokens, n_hidden, n_layers, starting_epoch = load_checkpoint(CHECKPOINT_PATH, model, opt)
        starting_epoch += 1
        print(f"Resuming {model_name} from epoch {starting_epoch+1} ...")
        model.to(device)

    else:
        starting_epoch = 0
        print(f"Starting to train {model_name}...")

    # train mode
    model.train()
    
    n_hidden = model.n_hidden
    n_layers = model.n_layers
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    model.to(device)
    
    tokens = len(model.chars)
    for e in range(starting_epoch, epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)
        for x, y in get_batches(data, batch_size, seq_length):
            
            # One-hot encode, make tensor move to cuda
            x = one_hot_encode(x, tokens)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
           
            inputs, targets = inputs.to(device), targets.to(device)

            # Creating a separate variables for the hidden state
            h = tuple([each.data for each in h])

            model.zero_grad()
           
            output, h = model(inputs, h)
            # some reshaping for output needed
            if bidirectional:
                output = output.view(output.size(0), output.size(2), output.size(1))
                loss = criterion(output, targets)
            else:
                loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            if resume_from_saved:
                model.cpu()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            # put back to cuda if was detached
            model.to(device)
            
            val_h = model.init_hidden(batch_size)
            val_losses = []

            move to eval mode
            model.eval()
            # for x, y in get_batches(val_data, batch_size, seq_length):

            #     # repest for validation
            #     x = one_hot_encode(x, n_chars)
            #     x, y = torch.from_numpy(x), torch.from_numpy(y)
                
            #     val_h = tuple([each.data for each in val_h])
                
            #     inputs, targets = x, y
                
            #     inputs, targets = inputs.cuda(), targets.cuda()

            #     output, val_h = model(inputs, val_h)
            #     if bidirectional:
            #         output = output.view(output.size(0), output.size(2), output.size(1))
            #         val_loss = criterion(output, targets)
            #     else:
            #         val_loss = criterion(output, targets.view(batch_size*seq_length).long())
            
            #     val_losses.append(val_loss.item())
            
            # model.train() # reset to train 

        save_checkpoint(model, opt, tokens, n_hidden, n_layers, e, CHECKPOINT_PATH)   
        print("Epoch: {}...".format(e+1),
            "Loss: {:.4f}...".format(loss.item()))
                # "Val Loss: {:.4f}".format(np.mean(val_losses)))