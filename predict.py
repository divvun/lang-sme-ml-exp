import torch
import torch.nn.functional as F
import numpy as np
from baseline_preprocess_input import one_hot_encode

def predict(net, char, device, h=None, top_k=None):
      
        # tensor inputs
        x = np.array([[net.char2int[char]]])
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

def show_sample(net, size, device, prime='The', top_k=None):
        

    net.to(device)
    
    net.eval() # eval mode
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, device, h, top_k=top_k)

    chars.append(char)
    
    #  get a new char
    for ii in range(size):
        char, h = predict(net, chars[-1], device, h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)