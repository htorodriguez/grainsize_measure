# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:16:43 2020

@author: hto_r
"""
from torch import optim

def optimizer (model, lr):
    """
    returns an adam optmizer from Pytorch
    """
    
    return optim.Adam(model.parameters(), lr)