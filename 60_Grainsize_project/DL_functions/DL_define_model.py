# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:00:05 2020

@author: hto_r
"""

import torch
from torchvision import datasets, transforms , models
from torch import nn, optim
import torch.nn.functional as F

def DL_model (HL_1, HL_2, Activation_function, dropout):
    """ Function to define a simple 2 hidden layer  Neural network
    args:
        HL_1: int with the dimension of the first hidden layer
        HL_2: int with the dimension of the second hidden layer
        Activation_function: string with an activation
                            function recognized by pytorch
        dropout: float, probability to dropout every neuron before each layer
    
    returns
        pytorch model 
    """
    
    device =torch.device ("cuda" if torch.cuda.is_available() else "cpu")
    
    if Activation_function =='relu':
            model=DL_model_relu_2HL(HL_1, HL_2, dropout)

    model.to(device)
    
    return(model)


def DL_model_relu_2HL (HL_1, HL_2, dropout=0.2):
    """ Function to define a simple 2 hidden layer  Neural network
    args:
        HL_1: int with the dimension of the first hidden layer
        HL_2: int with the dimension of the second hidden layer

        dropout: float, probability to dropout every neuron before each layer
    
    returns
        pytorch model 
    """
    
    device =torch.device ("cuda" if torch.cuda.is_available() else "cpu")
     
    class Grainsize_model (nn.Module):
        
        def __init__(self):
            super().__init__()
            self.fc1 =  nn.Linear(50**2*3, HL_1)
            self.fc2 =  nn.Linear(HL_1, HL_2)
            self.fc3 =  nn.Linear(HL_2, 100)
            self.fc4 =  nn.Linear(100, 4)
            self.Dropout = nn. Dropout(p=dropout)
        
        def forward (self, x):
            x= x.view(x.shape[0], -1)
            x= self.Dropout(F.relu(self.fc1(x)))
            x= self.Dropout(F.relu(self.fc2(x)))
            x= self.Dropout(F.relu(self.fc3(x)))
            x= F.log_softmax(self.fc4(x), dim=1)
       
            return  x
        
    model= Grainsize_model()
    model.to(device)
    
    return model