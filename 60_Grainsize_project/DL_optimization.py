# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:28:19 2020

@author: hto_r
"""
# =============================================================================
# GLobal imports
# =============================================================================
import torch
from torchvision import datasets, transforms , models
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Local imports
# =============================================================================
main_dir=os.getcwd()
os.chdir(main_dir)
    
from DL_functions import DL_define_model
from DL_functions import DL_transformation
from DL_functions import DL_generator
from DL_functions import DL_datafolder
from DL_functions import DL_optimizer
from DL_functions import DL_train
from DL_functions import DL_helper_functions
# =============================================================================
# =============================================================================
# # Main function
# =============================================================================
# =============================================================================

def import_train_vgg(train_dir='./DL_data/100_data/train', test_dir='./DL_data/100_data/test', epochs=1000):
    """
    Args: train and test image directories, where the images are sorted into 
          3 folders with names , 10, 100, and 1000 corresponding to the image
          classes
    Output: None. It saves a model checkpoint at ./DL_pretrained/checkpoint_vgg11.pth 
    
    """
    
    # =============================================================================
    # Load pretrained network
    # =============================================================================
    # load model strucutre
    model=models.vgg11(pretrained=True)
    # load weights
    # in case you have downloaded the model, because your firewall does not allow
    # to access the models weights use the command below
    #model.load_state_dict(torch.load('./DL_pretrained/vgg11-bbd30ac9.pth'))
    
    # =============================================================================
    # Define parameteres
    # =============================================================================
    PIL_brightness_1=[2]
    PIL_brightness_2=[1]
    Image_crop=[500]
    Resize_1=[244]
    Batch_size=[10]
    Activation_function=['relu']
    Dropout=[0.0]
    Lr=[1e-6]
    # =============================================================================
    # Make parameter variation
    # =============================================================================
    param_list=DL_helper_functions.make_param_list_5(PIL_brightness_1, 
                                                     Image_crop,
                                                     Resize_1,
                                                     Batch_size,
                                                     Lr)
    # =============================================================================
    # Training
    # =============================================================================
    
    for i, param in enumerate(param_list):
        
        #Associate parameters
        pil_brightness_1=param[0]
        image_crop=param[1]
        resize_1=param[2]
        batch_size=param[3]
        lr=param[4]
        #define the transformations
        train_transformation=DL_transformation.define_transformation(image_crop,
                                                                    resize_1,
                                                                    pil_brightness_1,
                                                                    pil_brightness_1)
        
        train_generator = DL_generator.define_generator(train_dir,
                                                      train_transformation,
                                                      batch_size)
    
        test_generator  = DL_generator.define_generator(test_dir,
                                                      train_transformation,
                                                      batch_size)  
        
        train_data      = DL_datafolder.define_datafolder(train_dir, train_transformation)
        
        #Set new layers to optimize
        
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False
        #Freeze all layers to no grad
        set_parameter_requires_grad(model, True) 
        # expand the net
        model.classifier[6] = nn.Sequential (
                                            nn.Linear(4096, 256),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.2),
                                            nn.Linear(256, 3),# set the total 
                                            nn.LogSoftmax(dim=1)
                                            ) 
        
        #Extract parameters to vary and set the optmizernad loss criterion
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        model=model.to(device)
        feature_extract = True
        params_to_update=model.parameters()    
        
        if feature_extract:
            params_to_update=[]
            for name, param in model.named_parameters():
                if param.requires_grad ==True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad ==True:
                    print("\t", name)
        
        optimizer=optim.Adam(params_to_update, lr)
        criterion = nn.NLLLoss()
    
        # train and validate
        try:
            (train_losses, validation_losses, accuracy) = DL_train.train(
                                                                        train_generator, 
                                                                        test_generator,
                                                                        criterion, 
                                                                        model, 
                                                                        epochs, 
                                                                        optimizer, 
                                                                        batch_size
                                                                        )
        
            output_df=pd.DataFrame({
                    'train_losses' : train_losses,
                    'validation_losses': validation_losses,
                    'accuracy': accuracy
                    })
        
            output_df['parameters']=str(param)
            
            checkpoint={
                    'input_size': 244,
                    'criterion': 'NLLLoss',
                    'optimizer_state': optimizer.state_dict,
                    'state_dict':model.state_dict(),
                    'epochs' : epochs,
                    'Mapping_class_idx_train': train_data.class_to_idx
                    }
            torch.save(checkpoint, './DL_pretrained/checkpoint_vgg11.pth')
            
        except:
            print('Following parameters were not succesfull: ', param) 
            
        return(output_df)