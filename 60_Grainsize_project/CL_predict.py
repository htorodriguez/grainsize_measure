# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 12:50:47 2020

@author: hto_r
"""
# =============================================================================
# GLobal imports
# =============================================================================

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import math
# =============================================================================
# Local imports
# =============================================================================

main_dir= os.getcwd()
os.chdir(main_dir)

from DL_functions import DL_helper_functions
from CL_functions import CL_DBscan
from CL_functions import CL_load

# =============================================================================
# =============================================================================
# # Main function
# =============================================================================
# =============================================================================

def get_CL_parameters(file_pointer, class_10_100_1000):
    """ Function to predict cluster count and mean size by means of clustering
    Args: 
        file_pointer: string with a file path
    
    Returns
        tuple with(
                clusters: predicted number of clusters
                log_mean_radius: predicted mean radius in log (pixels)
    """
    
    #Constant parameters

    
    # =========================================================================
    # optimal parameters 
    # =========================================================================
    #for imagge class 1000
    binarize_list=[70]
    eps_list=[2]
    min_samples=[2]
    filter_boundary_list=[100]
    remove_large_clusters_list=[200]
    remove_small_clusters_list=[10]
    max_filter_list=[3]
    eps_grain_boundary_list=[2]
    min_sample_grain_boundary_list=[5]
    binarize_bdr_coord_list=[100]
    binarize_grain_coord_list=[100]
    
    param_list_1000=DL_helper_functions.make_param_list_11(
                                                     binarize_list,
                                                     eps_list,
                                                     min_samples,
                                                     filter_boundary_list,
                                                     remove_large_clusters_list,
                                                     remove_small_clusters_list,
                                                     max_filter_list,
                                                     eps_grain_boundary_list,
                                                     min_sample_grain_boundary_list,
                                                     binarize_bdr_coord_list,
                                                     binarize_grain_coord_list,

                                                     ) 
    
    # for image class 100 objects
    
    binarize_list=[15]
    eps_list=[2]
    min_samples=[2]
    filter_boundary_list=[100]
    remove_large_clusters_list=[20]
    remove_small_clusters_list=[100]
    max_filter_list=[3]
    eps_grain_boundary_list=[2]
    min_sample_grain_boundary_list=[5]
    binarize_bdr_coord_list=[100]
    binarize_grain_coord_list=[100]
    
    param_list_100=DL_helper_functions.make_param_list_11(
                                                     binarize_list,
                                                     eps_list,
                                                     min_samples,
                                                     filter_boundary_list,
                                                     remove_large_clusters_list,
                                                     remove_small_clusters_list,
                                                     max_filter_list,
                                                     eps_grain_boundary_list,
                                                     min_sample_grain_boundary_list,
                                                     binarize_bdr_coord_list,
                                                     binarize_grain_coord_list,

                                                     ) 
    
    # for image class 10 objects
    binarize_list=[30]
    eps_list=[2]
    min_samples=[5]
    filter_boundary_list=[100]
    remove_large_clusters_list=[0]
    remove_small_clusters_list=[800]
    max_filter_list=[3]
    eps_grain_boundary_list=[2]
    min_sample_grain_boundary_list=[5]
    binarize_bdr_coord_list=[100]
    binarize_grain_coord_list=[100]
    
    param_list_10_1=DL_helper_functions.make_param_list_11(
                                                     binarize_list,
                                                     eps_list,
                                                     min_samples,
                                                     filter_boundary_list,
                                                     remove_large_clusters_list,
                                                     remove_small_clusters_list,
                                                     max_filter_list,
                                                     eps_grain_boundary_list,
                                                     min_sample_grain_boundary_list,
                                                     binarize_bdr_coord_list,
                                                     binarize_grain_coord_list,

                                                     ) 
    
    
    binarize_list=[5]
    eps_list=[2]
    min_samples=[5]
    filter_boundary_list=[100]
    remove_large_clusters_list=[0]
    remove_small_clusters_list=[800]
    max_filter_list=[3]
    eps_grain_boundary_list=[2]
    min_sample_grain_boundary_list=[5]
    binarize_bdr_coord_list=[100]
    binarize_grain_coord_list=[100] 
    
    param_list_10_2=DL_helper_functions.make_param_list_11(
                                                     binarize_list,
                                                     eps_list,
                                                     min_samples,
                                                     filter_boundary_list,
                                                     remove_large_clusters_list,
                                                     remove_small_clusters_list,
                                                     max_filter_list,
                                                     eps_grain_boundary_list,
                                                     min_sample_grain_boundary_list,
                                                     binarize_bdr_coord_list,
                                                     binarize_grain_coord_list,

                                                     ) 

    # =========================================================================
    # optimal parameters 
    # =========================================================================

        
    if class_10_100_1000 =='10_1':
        param=param_list_10_1[0]
    if class_10_100_1000 =='10_2':
        param=param_list_10_2[0]
    if class_10_100_1000 =='100':
        param=param_list_100[0]
    if class_10_100_1000 =='1000':
        param=param_list_1000[0]
    
    #define parameters
    binarize=param[0]
    eps=param[1]
    min_sample=param[2]
    filter_boundary=param[3]
    remove_large_clusters=param[4]
    remove_small_clusters=param[5]
    max_filter=param[6]
    eps_grain_boundary=param[7]
    min_sample_grain_boundary=param[8]
    binarize_bdr_coord=param[9]
    binarize_grain_coord=param[10]
    
    (image_X, m_real, s_real)=CL_load.load_func(file_pointer, threshold=binarize, max_filter=3)
    try:
        print('Clustering image')
        (m_CL, s_CL, clusters) = CL_DBscan.fit_DBscan(image_X, 
                                                      eps,
                                                      eps_grain_boundary,
                                                      min_sample,
                                                      min_sample_grain_boundary,
                                                      filter_boundary, 
                                                      remove_large_clusters, 
                                                      remove_small_clusters,
                                                      binarize_bdr_coord,
                                                      binarize_grain_coord,
                                                          )
        if math.isnan(m_CL):
            (m_CL,clusters)=(0,0)
            
    except:
        print('fit went wrong', str(param)) 
        (m_CL,clusters)=(0,0)
        
    
    log_mean_radius=m_CL
    #print(m_CL)
    
    return(clusters, log_mean_radius)

