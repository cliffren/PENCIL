import numpy as np
from numpy.core.arrayprint import repr_format
import torch

def zero_one_rejection_loss(y, h, r=1, c=0):
    '''count loss.'''
    if type(r)!=type(1):
        r=np.array(r)
    l_total=np.mean(np.sum(((y*h)<=0)*(r>0)+c*(r<=0)))
    l_rejection=np.mean(np.sum(c*(r<=0)))
    return l_total, l_rejection

def binary_loss(y_true, y_pred, mtype='hinge', pthr=0.1):
    assert y_true.min() == -1 or y_true.min() == 1,  'label must be in {1, -1}.'
    if mtype == 'hinge':
        return torch.relu(1 - y_true * y_pred)
    elif mtype == 'soft-hinge':
        return 1 / (1 + y_true * y_pred)
    elif mtype == 'neg-product':
        return - y_true * y_pred
    elif mtype == 'ce':
        y_true = y_true / 2 + 0.5
        py_pred = 1 / (1 + torch.exp(-y_pred))
        return - y_true * torch.log(py_pred) - (1 - y_true) * torch.log(1 - py_pred)
    elif mtype == 'clip-ce':
        y_true = y_true / 2 + 0.5
        py_pred = 1 / (1 + torch.exp(-y_pred))
        return - y_true * torch.log(torch.relu(py_pred - pthr) + pthr) - (1 - y_true) * torch.log( torch.relu(1 - py_pred - pthr) + pthr)

def rejection_loss(e, r, c, mtype):  
    rej_loss_func_dict = {
        'LR': lambda e, r : torch.relu(torch.max(1 + (r + e) / 2, c * (1 - (1/(1-2*c))* r))),
        'GLR': lambda e, r : torch.relu(torch.max(r + e,c * (1 - r))),
        'NGLR': lambda e, r : torch.relu(torch.max(e * (1 + r), c * (1 - r))),
        'Sigmoid': lambda e, r: e * torch.sigmoid(r) + c * torch.sigmoid(-r) 
    }
    return rej_loss_func_dict[mtype](e, r)

def pseudo_mse_loss(pred, target, reduction='done'):
    a = 10.0
    b = 10.0
    x = pred - target
    factor = 2 / (1 + np.exp(a))
    l = torch.exp(b * x - a) / (1 + torch.exp(b * x - a))  + 1 / (1 + torch.exp(b * x + a))
    l = (l - factor) / (1 - factor) #scale to [0, 1)
    if reduction == 'none':
        return l
    else:
        return torch.mean(l)
