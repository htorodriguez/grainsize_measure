# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:48:10 2020

@author: hto_r
"""

import torchvision
import PIL
import numpy as np

def define_transformation (Image_crop, resize_1, brightness_1, brightness_2):
    """ Function defining an image transformation 
    Args: 
            Image crop: size to crop the image 
            resize_1: first resizing step
            brightness_1: parameter to apply to image jitter
            brightness_2: parameter to apply to image jitter
    
    returns a custom PIL transformation
    """

    train_transformation = torchvision.transforms.Compose([
            custom_transform(),
            #torchvision.transforms.RandomRotation(90),
            torchvision.transforms.RandomCrop(Image_crop), 
            torchvision.transforms.Resize(resize_1) ,
            torchvision.transforms.ColorJitter(brightness=(brightness_1, brightness_2),
                                               contrast=0, saturation=0, hue=0) ,
            
            torchvision.transforms.Resize (50) ,
            
            #torchvision.transforms.ColorJitter(brightness=(brightness_1, brightness_2),
            #                                   contrast=0, saturation=0, hue=0) ,
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor() ,
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),

            
            ])
    
    return train_transformation

def define_transformation_nonorm (Image_crop, resize_1, brightness_1, brightness_2):
    """ Function defining an image transformation, without norming the image 
    Args: 
            Image crop: size to crop the image 
            resize_1: first resizing step
            brightness_1: parameter to apply to image jitter
            brightness_2: parameter to apply to image jitter
    
    returns a custom PIL transformation
    """

    train_transformation = torchvision.transforms.Compose([
            custom_transform(),
            #torchvision.transforms.RandomRotation(90),
            torchvision.transforms.RandomCrop(Image_crop), 
            torchvision.transforms.Resize(resize_1) ,
            torchvision.transforms.ColorJitter(brightness=(brightness_1, brightness_2),
                                               contrast=0, saturation=0, hue=0) ,
            
            torchvision.transforms.Resize (50) ,
            
            #torchvision.transforms.ColorJitter(brightness=(brightness_1, brightness_2),
            #                                   contrast=0, saturation=0, hue=0) ,
            torchvision.transforms.RandomHorizontalFlip(),

            
            ])
    
    return train_transformation

# =============================================================================
# 
# =============================================================================


class custom_transform(object):
    """ Class defining a custom image transformation to make an RGB image Grey
        and normalize
    PIL image input and output after operations
    """
    
    def __call__(self, img):
        
        """
        
        """
        img_array=np.array(img)
        img_array_r=img_array[:,:,0]
        img_array_g=img_array[:,:,1]
        img_array_b=img_array[:,:,2]
        img_array_grey=img_array_r + img_array_g + img_array_b
        img_array_grey=img_array_grey/3
        img_array_grey=img_array_grey/img_array_grey.max()*255
        img_array_grey=img_array_grey.astype(np.uint8)
        t=np.zeros((500,501))
        t[:,:-1]=img_array_grey
        img_array_grey=np.diff(t)
        img_array_grey=binarize_v(img_array_grey)
        img_array_grey=img_array_grey/img_array_grey.max()*255
        img_array_grey=img_array_grey.astype(np.uint8)
        
        img_array[:,:,0]=img_array_grey
        img_array[:,:,1]=img_array_grey
        img_array[:,:,2]=img_array_grey  
        res= PIL.Image.fromarray(img_array.astype('uint8'), 'RGB')
        return res
# =============================================================================
# Helper numpy array transformation functions
# =============================================================================

def binarize (x):
    """Function to apply to a pixel value to make a threshold operation
    args : int
    
    returns int equal to 0 or 255
    """
    threshold_1=-10
    threshold_2= 10
    
    if (x <threshold_1) |(x >threshold_2) :
        return 255
    else:
        return 0

binarize_v= np.vectorize(binarize)
# =============================================================================
# 
# =============================================================================

def threshold (x):
    """Function to apply to a pixel value to make a threshold operation
    args : int
    
    returns int equal to 0 or 255
    """
    threshold_1=0
    
    if (x <threshold_1):
        return 0
    else:
        return 255

threshold_v= np.vectorize(binarize)
# =============================================================================
# 
# =============================================================================

def lump (A):
    """Function to apply to a  numpy array of an image, to average its values
    args : numpa array
    
    returns average numpy array B
    """
    B=np.zeros((A.shape[0], A.shape[1]))
    for i in range(1, A.shape[0]-2, 1):
        for j in range(1, A.shape[1]-2, 1):
            
            m=A[i-1:i+1, j-1:j+1].mean()
            
            B[i, j]=m
            B[i-1, j]=m
            B[i, j-1]=m
            B[i-1, j-1]=m
            B[i+1, j+1]=m
    
    return B