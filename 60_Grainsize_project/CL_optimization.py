# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 19:57:13 2020

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
# =============================================================================
# Local imports
# =============================================================================

main_dir= os.getcwd()
os.chdir(main_dir)

from DL_functions import DL_helper_functions
from CL_functions import CL_DBscan
from CL_functions import CL_load
# =============================================================================
# define directory and paramters
# =============================================================================

def CL_optimize(datafolder):
    """ Function to perform grid search to determine optimal clustering 
    parameters
    
    Args:datafolder
        The images within the folder must be labelled as in the following label
                1pepper_m_1.6_s_1.5.bmp
        where 
        the value after m corresponds to the log10(mean size in pixel)
        the value after s corresponds to the log10(object counts)
        
    Returns: 
        dataframe: containing filename, parameters, labelled m and s, predicted m and s
        class of the image, predicted class

    """
    file_directory=main_dir+ '\\DL_data\\200_predict'
    
    binarize_list=[2,20]
    eps_list=[2, 4]
    min_samples=[1, 5]
    filter_boundary_list=[50, 200]
    remove_large_clusters_list=[0, 10]
    remove_small_clusters_list=[1, 1000]
    
    max_filter_list=[1,5]
    eps_grain_boundary_list=[1, 4]
    min_sample_grain_boundary_list=[3, 7]
    binarize_bdr_coord_list=[80,120]
    binarize_grain_coord_list=[80, 120]
    
    
    # =============================================================================
    param_list=DL_helper_functions.make_param_list_11(
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
    # =============================================================================
    # Grid search optmization
    # =============================================================================
                                                        
    output_df_total=pd.DataFrame()
    
    for i , param in enumerate(param_list):
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
        #initialize output array
        cluster_list=[]
        m_real_list=[]
        s_real_list=[]
        m_predict=[]
        s_predict=[]
        file_list=os.listdir(file_directory)
        file_list=[file for file in file_list if ('bmp' in file)]
        #load image
        for file in file_list:
            print('load image', file)
            (image_X, m_real, s_real)=CL_load.load_func(file_directory+'\\'+file, threshold=binarize, max_filter=max_filter)
            try:
                print('Clustering image')
                (m_CL, s_CL, clusters) = CL_DBscan.fit_DBscan(
                                                                image_X, 
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
                
                print('There are {} clusters'.format(clusters))
                print('with a mean pixel radius of {}'.format(round(10**m_CL)))
                cluster_list.append(clusters)
                m_predict.append(m_CL)
                s_predict.append(s_CL)
                m_real_list.append(m_real)
                s_real_list.append(s_real)
                
            except:
                print('fit went wrong', str(param))
                
        
        output_df=pd.DataFrame({
                'file':file,
                'clusters':cluster_list,
                'm_real': m_real_list,
                'm_predict':m_predict,
                's_real':s_real_list,
                's_predict':s_predict
                })
        output_df['parameters']=str(param)
        output_df_total=pd.concat([output_df_total, output_df])
    
    output_df_total['diff_m']=abs(output_df_total['m_real'].astype('float')-output_df_total['m_predict'])
    output_df_total['diff_s']=abs(output_df_total['s_real'].astype('float')-output_df_total['s_predict'])

    return(output_df_total)
# =============================================================================
# 
# =============================================================================
