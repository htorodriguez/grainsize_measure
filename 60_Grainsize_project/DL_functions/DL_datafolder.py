# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:23:28 2020

@author: hto_r
"""

import torchvision

def define_datafolder(data_dir, transformation):
    """ Function to define a pytorch Datafolder
    args:
        data_dir: string with the path to the data folder, that contains two
                 two folder with train and test data
    returns
        pytorch datafolder
    """
    
    data= torchvision.datasets.ImageFolder(root=data_dir, transform=transformation)
    
    return data