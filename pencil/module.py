import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from mlflow import log_params

class PredNet(nn.Module):
    """
    Classification Network: Should have more capacity than the classification net
    """
    def __init__(self, in_features, out_features=1, hide_features=[10], model_type='linear', dropout=0., mlflow_record=True):
        super(PredNet, self).__init__()
        if model_type == 'linear':
            hide_features = None
            
        if mlflow_record:
            log_params({
                'class_net':{
                    'in_features': in_features,
                    'out_features': out_features,
                    'hide_features': hide_features,
                    'model_type': model_type
                }
            })
        
        if type(hide_features) is not list:
            hide_features = [hide_features]
            
        self.select_weight = nn.Parameter(torch.rand(in_features) / np.sqrt(in_features))
        
        if model_type == 'linear':
            self.out = nn.Linear(in_features, out_features)
        elif model_type == 'non-linear':
            self.hidden_layers = nn.ModuleList([nn.Linear(in_features, hide_features[0])])
            for i in range(len(hide_features) - 1):
                self.hidden_layers.append(nn.Linear(hide_features[i], hide_features[i+1]))
            # self.hidden1 = nn.Linear(in_features, hide_features)
            self.dropout = nn.Dropout(p=dropout)
            self.out = nn.Linear(hide_features[-1], out_features)
        
        self.model_type = model_type

    def forward(self,x):
        x = x * self.select_weight
        if self.model_type == 'linear':
            x = self.out(x)
            # x = F.dropout(x, 0.2)
        elif self.model_type == 'non-linear':
            for layer in self.hidden_layers: 
                x = layer(x)
                #x = torch.tanh(x)
                x = self.dropout(x)
                x = x*(torch.tanh(F.softplus(x)))
            # x = self.hidden1(x)
            #x = torch.tanh(x)
            x = self.dropout(x)
            x = x*(torch.tanh(F.softplus(x))) #MISH:  (https://arxiv.org/abs/1908.08681v1)
            
            x = self.out(x)
        return x


class RejNet(nn.Module):
    """
    Rejection Network: Should have more capacity than the classification net
    """
    def __init__(self, in_features, hide_features=[200], model_type='linear', tanh=False, dropout=0., mlflow_record=True):
        super(RejNet, self).__init__()
        if model_type == 'linear':
            hide_features = None
        
        if mlflow_record:    
            log_params({
                'rej_net':{
                    'in_features': in_features,
                    'hide_features': hide_features,
                    'model_type': model_type,
                    'tanh': tanh
                }
            })

        if type(hide_features) is not list:
            hide_features = [hide_features]
            
        self.model_type = model_type
        if self.model_type == 'non-linear':   
            self.hidden_layers = nn.ModuleList([nn.Linear(in_features, hide_features[0])])
            for i in range(len(hide_features) - 1):
                self.hidden_layers.append(nn.Linear(hide_features[i], hide_features[i+1]))

            # self.hidden2 = nn.Linear(hide_features, hide_features)
            self.dropout = nn.Dropout(dropout)
            self.out = nn.Linear(hide_features[-1], 1)
        elif self.model_type == 'linear':
            self.out = nn.Linear(in_features, 1)
        
        self.tanh = tanh

    def forward(self, x):
        if self.model_type == 'non-linear':
            for layer in self.hidden_layers: 
                x = layer(x)
                #x = torch.tanh(x)
                x = self.dropout(x)
                x = x*(torch.tanh(F.softplus(x)))
                
            x = self.out(x)
            if self.tanh:
                x = torch.tanh(x)
        elif self.model_type == 'linear':
            x = self.out(x)
            if self.tanh:
                x = torch.tanh(x)
        return x
    
class GSlayer(nn.Module):
    def __init__(self, in_features):
        super(GSlayer, self).__init__()
        self.select_weight = nn.Parameter(torch.rand(in_features) / np.sqrt(in_features))
        
    def forward(self, x):
        x = x * self.select_weight
        return x

class PencilModel(nn.Module):
    def __init__(self, predictor, rejector, gslayer=None, mlflow_record=True):
        super(PencilModel, self).__init__()
        
        self.predictor = predictor
        self.rejector = rejector
        self.gslayer = gslayer
        
        if mlflow_record:
            if gslayer is None:
                log_params({
                    'gslayer': 'none'
                })
            else:
                log_params({
                    'gslayer': 'not-none'
                })
    
    def forward(self, x):
        if self.gslayer is not None:
            x = self.gslayer(x)
        h = self.predictor(x)
        r = self.rejector(x)
        
        return h, r
    

        