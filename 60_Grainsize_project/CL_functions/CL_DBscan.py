# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:14:45 2020

@author: hto_r
"""

# =============================================================================
# GLobal imports
# =============================================================================
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# =============================================================================
# Main Clustering function
# =============================================================================

def fit_DBscan (image_X, 
                eps,
                eps_grain_boundary,
                min_sample,
                min_sample_grain_boundary,
                filter_boundary, 
                remove_large_clusters, 
                remove_small_clusters,
                binarize_bdr_coord,
                binarize_grain_coord,
                ):
    """ Function to measure counts and average sizes of instances within an image
    args:
        image_X: np array containing a preporcessed image 
        eps: float, parameter for the DBscan algorithm from sklearn
        eps_grain_boundary:float, parameter for the DBscan algorithm from sklearn
        min_sample: int, parameter for the DBscan algorithm from sklearn
        min_sample_grain_boundary: int float, parameter for the DBscan algorithm from sklearn
        filter_boundary:int, threshold to apply while finding the grain boundaries 
        remove_large_clusters: int indicating how many of the largest clusters
                                shall be removed
        remove_small_clusters:int indicating how many of the smallest clusters
                                shall be removed
        binarize_bdr_coord: int for the binarization of the grain boundaries
        binarize_grain_coord:: int for the binarization of the grain interiors
    
    returns:
        m_CL: float, log10(mean (predicted cluster radius in pixels))
        s_CL: float, log10(predicted cluster count)
    """
    print('Finding grain boundaries')
    bdr_coord=np.array(binarize_array_high(image_X, binarize_bdr_coord))
    bdr_coord=find_grain_boundaries(bdr_coord, eps=eps_grain_boundary, min_sample=min_sample_grain_boundary, filter_boundary=filter_boundary)
    bdr_coord_df=pd.DataFrame(bdr_coord)
    bdr_coord_df.columns=['X','Y']
    bdr_coord_df['Z']=255
    df_grain=pd.pivot_table(bdr_coord_df, index='X', columns='Y', values='Z', fill_value=0)
    df_grain=df_grain.to_numpy()
    
    print('Measuring grains')
    grain_coord = np.array(binarize_array(df_grain, binarize_grain_coord))
    (m_CL, s_CL, clusters)=find_grains(grain_coord, eps, min_sample, remove_large_clusters, remove_small_clusters)
    return (m_CL, s_CL, clusters)

# =============================================================================
# Subfunction of the clustering algortithm
# =============================================================================
def find_grain_boundaries(X_coord, eps=1, min_sample=1, filter_boundary=100):
    """Function to find the grain boundaries within a list of x,y coordinates
    args:
        X_coord: list of (x,y) coordinates 
        eps: float, parameter for the DBscan algorithm from sklearn
        min_sample:int, parameter for the DBscan algorithm from sklearn
        filter_boundary:int, threshold to apply while finding the grain boundaries
    returns:
        grain_boundary_list: list of (x,y) coordinates of the predicted 
                            grain boundaries
        
    """
    
    clustering=DBSCAN(eps=eps, min_samples=min_sample, metric='euclidean').fit(X_coord)
    
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_]=True
    
    labels = clustering.labels_
    
    #get legths from df
    lengths_df= get_length_labels(labels)
    lengths_df=lengths_df[lengths_df.length>filter_boundary]
    #remove noise label -1
    lengths_df=lengths_df[lengths_df.label>-1]
    unique_labels=list(lengths_df.label)
    #
    
    grain_boundary_df = pd.DataFrame()
    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = X_coord[class_member_mask & core_samples_mask]
        grain_boundary_df= grain_boundary_df.append(pd.DataFrame(xy))
    
    grain_boundary_df=grain_boundary_df.reset_index(drop=True)
    grain_boundary_list=grain_boundary_df.to_numpy()
    return (grain_boundary_list)

# =============================================================================
# 
# =============================================================================
def find_grains(grain_coord, eps, min_sample, remove_large_clusters, remove_small_clusters):
    """Function to find the grain within a list of (x,y) coordinates of all grains
    args:
        grain_coord: list of (x,y) coordinates of all grains
        eps: float, parameter for the DBscan algorithm from sklearn
        min_sample:int, parameter for the DBscan algorithm from sklearn
        remove_large_clusters:int indicating how many of the largest clusters
                                shall be removed
        remove_small_clusters:int indicating how many of the smallest clusters
                                shall be removed
    returns:
        log10(mean (predicted cluster radius in pixels))
        log10(standard deviation (predicted cluster radius in pixels))
        predicted cluster count
        
    """
    clustering=DBSCAN(eps=eps, min_samples=min_sample, metric='euclidean').fit(grain_coord)
    
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_]=True
    
    labels = clustering.labels_
    #get legths from df
    lengths_df= get_length_labels(labels)

    #remove noise label -1
    lengths_df=lengths_df[lengths_df.label>-1]
    #remove too small clusters according to paramters
    lengths_df=lengths_df[lengths_df.length>remove_small_clusters]
    #Remove non-sense clsuters
    lengths_df=lengths_df[lengths_df.length<200*200]
    #Remove large cluster according to the optmiization
    
    if remove_large_clusters>0:
        lengths_df=lengths_df.iloc[:-1*remove_large_clusters, :]
    
    unique_labels=list(lengths_df.label)
    # 
    lengths_dilated=[]
    #plt.figure()
    for k in unique_labels:   
        class_member_mask = (labels == k)
        #Get the cooridnates of the given grain
        xy = grain_coord[class_member_mask & core_samples_mask]
        #Transform to df and pivot
        xy_coord_df=pd.DataFrame(xy)
        xy_coord_df.columns=['X','Y']
        xy_coord_df['Z']=255
        #Get the rigini coordinates of a single grain
        (x_min, y_min)=( xy_coord_df.X.min(), xy_coord_df.Y.min() )
        df_grain=pd.pivot_table(xy_coord_df, index='X', columns='Y', values='Z', fill_value=0)
        df_grain=df_grain.to_numpy()
        #transform to image
        img = Image.fromarray(df_grain.astype(np.uint8))
        #dilation
        for i in range(3):
            img= img.filter(ImageFilter.MaxFilter(3))
        #transform back to array
        df_grain_dilated=np.array(img)
        #add the orign of the single grains
        current_grain_coord = np.array(binarize_array_high(df_grain_dilated, 100))
        current_grain_coord_df = pd.DataFrame(current_grain_coord, columns =['X','Y'])
        current_grain_coord_df.X=current_grain_coord_df.X + x_min
        current_grain_coord_df.Y=current_grain_coord_df.Y + y_min
        current_grain_coord =current_grain_coord_df.to_numpy()
        #
        xy = current_grain_coord
        #
        lengths_dilated.append(len(xy))
        #####plot
        #y is flipped to correspond to the image
        #plt.plot(xy[:,1], 500-xy[:,0], 'o', markersize=0.5)
    
    #plt.show()
    ##########
       
    radii_dilated=[(a/3.1416)**0.5 for a in lengths_dilated]
    log_radii_dbscan=[np.log10(r) for r in radii_dilated]
    clusters=len(radii_dilated)
    
    return(np.array(log_radii_dbscan).mean(), np.array(log_radii_dbscan).std(), clusters)
    
# =============================================================================
#  Helper Functions   
# =============================================================================
def binarize_array(img, limit):
    """Function to get the coordinates of an image array containing a value
        below a predetermined limit
    args:
        img: numpy array
        limit: int from 0 to 255
    returns:
        coord: list of tuples containing the (x,y) coordinates
    """
    coord=[]
    for  x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x,y]<limit:
                coord.append((x,y))
            else:
                pass
    return coord
# =============================================================================
# 
# =============================================================================
def binarize_array_high(img, limit):
    """Function to get the coordinates of an image array containing a value
        above a predetermined limit
    args:
        img: numpy array
        limit: int from 0 to 255
    returns:
        coord: list of tuples containing the (x,y) coordinates
    """
    coord=[]
    for  x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x,y]>limit:
                coord.append((x,y))
            else:
                pass
    return coord

# =============================================================================
# 
# =============================================================================
def get_length_labels(labels):
    """Function to obtain the numer of different labels, and their length from
        a label list, as is obtained from a clustering algorithm
    args: 
        labels: list of labels
    return: 
        df with 
            label, 
            lengths, which is equivalent to the area when the clusters
            have 2 dimensions
            radii
    """
    unique_labels = set(labels)
    label=[]
    lengths=[]
    radius=[]
    for k in list(unique_labels):
        label.append(k)
        lengths.append( (labels == k ).sum())
        radius.append( ( (labels == k ).sum()/3.1416)**0.5 )
    
    lengths_df = pd.DataFrame({
                                'label':label,
                                'length':lengths,
                                'radius':radius
            
                                })
    return lengths_df
    
# =============================================================================
# 
# =============================================================================