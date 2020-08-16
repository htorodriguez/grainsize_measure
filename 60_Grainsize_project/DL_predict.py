# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:20:11 2020

@author: hto_r
"""
# =============================================================================
# GLobal imports
# =============================================================================
import os 
import torch
from torchvision import datasets, transforms , models
import torchvision.transforms.functional as F
import PIL
import numpy as np
from IPython.display import display
# =============================================================================
# Local imports
# =============================================================================
main_dir=os.getcwd()
os.chdir(main_dir)
from DL_functions import DL_define_model
from DL_functions import DL_transformation
from DL_functions import DL_generator
from DL_functions import DL_helper_functions

# =============================================================================
# =============================================================================
# Main function
# =============================================================================
# =============================================================================
def predict_DL(file_path):
    """Function to load and initialize the modified VGGnet-11 Neural network
    Args:       string with the file_path
    
    Returns:    Predicted class
    """
    
    (model,param_list, train_transformation, idx_to_class_dict)= load_and_initialize_DL()
    
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_class = evaluate_model(file_path, model,param_list, train_transformation, idx_to_class_dict)
    
    return(result_class)
# =============================================================================
# =============================================================================
# Sub functions
# =============================================================================
# =============================================================================
def load_and_initialize_DL():
    """Function to load and initialize the modified VGGnet-11 Neural network
    Args:       none
    
    Returns:    tuple of 
                model, 
                param_list, 
                train_transformation, 
                idx_to_class_dict
    """

    # =============================================================================
    # Load pretrained network
    # =============================================================================
    # load model structure
    # The model used was the vgg11
    model=models.vgg11()
    # Add the final layer, as was trained
    model.classifier[6] = torch.nn.Sequential(  torch.nn.Linear(4096, 256),
                                                torch.nn.ReLU(inplace=True),
                                                torch.nn.Dropout(0.2),
                                                torch.nn.Linear(256, 3),# set the total 
                                                torch.nn.LogSoftmax(dim=1)
                                                ) 
    # load weights
    checkpoint= torch.load(main_dir+'\\DL_pretrained\checkpoint_vgg11.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # Get the idx to class mapping
    idx_to_class_dict = {v: k for k, v in checkpoint['Mapping_class_idx_train'].items()}
    # =============================================================================
    # Set the constant paramters, not to be chagned these are already optmized
    # =============================================================================
    PIL_brightness_1=[2]
    Image_crop=[500]
    Resize_1=[244]
    Batch_size=[10]
    Lr=[1e-6]
    # =============================================================================
    # Make parameter variation list
    # =============================================================================
    param_list=DL_helper_functions.make_param_list_5(PIL_brightness_1, 
                                                     Image_crop,
                                                     Resize_1,
                                                     Batch_size,
                                                     Lr)
 
    #define constants
    pil_brightness_1=param_list[0][0]
    image_crop=param_list[0][1]
    resize_1=param_list[0][2]
    
    #define the transformations
    train_transformation=DL_transformation.define_transformation(image_crop,
                                                                 resize_1,
                                                                 pil_brightness_1,
                                                                 pil_brightness_1)

    return (model,param_list, train_transformation, idx_to_class_dict)
 
# =============================================================================
# =============================================================================
def evaluate_model(file_path, model,param_list, train_transformation, idx_to_class_dict):
    """Function evaluate an image based on the modified VGGnet-11 Neural network
    Args:      tuple of 
                model, 
                param_list, 
                train_transformation, 
                idx_to_class_dict
    
    Returns:    Predicted class
    """

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        
    im = PIL.Image.open(file_path)
    #use the transformation as is
    im_tensor=train_transformation(im)
    # add a dimension to simulate a batch
    im_tensor=im_tensor.unsqueeze(0)
    #predict
    with torch.no_grad():
        log_ps=model(im_tensor)
        ps=torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
    result_class=idx_to_class_dict[top_class.numpy()[0][0]]
    # print(ps)
    print('Cluster class ', result_class)
    
    return (result_class)

 # =============================================================================
# =============================================================================
# #    
