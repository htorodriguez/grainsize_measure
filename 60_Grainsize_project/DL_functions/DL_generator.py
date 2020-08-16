# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:48:10 2020

@author: hto_r
"""

import torch
import torchvision

def define_generator(data_dir, transformation, Batch_size):
    """ Function to define a pytorch image generator
    args:
        data_dir: string to the image data
        transformation: pytorch image transformation
        Batch_size: int with the batch size for the forward passes over the 
                    network
    
    returns:
        Pytorch image generator
    """
    
    data= torchvision.datasets.ImageFolder(root=data_dir, transform=transformation)
    
    data_generator=torch.utils.data.DataLoader(data, batch_size = Batch_size, shuffle=True)
    
    return data_generator