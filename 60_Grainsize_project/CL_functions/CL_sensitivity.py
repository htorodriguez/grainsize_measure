# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 17:33:54 2020
Functions for the sensitivity analyses
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

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import itertools

# =============================================================================
# Multivariate metamodel definition
# =============================================================================

def build_model():
    """
    Input: None
    Output: a sklearn model pipeline
    """
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(interaction_only=True)),
        ('multvariate_regression', LinearRegression())
        ])

    model_pipeline = pipeline

    return model_pipeline
# =============================================================================
# Build an fit model
# =============================================================================

def build_and_fit_model(df):
    """
    Input: Dataframe df with the input of the datamodel (p1 to p10) 
           and the output (clusters)
           
    Output: a trained meta-model 
    """
    parameter_name=['p1', 'p2', 'p3', 'p4','p5', 'p6','p7','p8', 'p9','p10', 'p11']
    X=df[parameter_name]
    Y=df['clusters']
    model=build_model()
    reg=model.fit(X,Y)

    reg = LinearRegression().fit(X, Y)
    print('Multivariate regression R2', reg.score(X, Y))

    return(reg)
# =============================================================================
# Define functions for sensitivity input
# =============================================================================
def my_func(oneD_array):
    """
    input np array with shape (1,x)
    ouput numerical result
    """
    pred=reg.predict(oneD_array)[0]
    return(pred)
    
def test_func(oneD_array):
    """
    input np array with shape (1,x)
    ouput numerical result
    """
    res=0
    for i in range(oneD_array.shape[1]):
        res+=oneD_array[:,i][0]
    return(res)
# =============================================================================
# Make parameter list variation for sensitivity analyses
# =============================================================================
def make_permutation_list(list_lists):
    """
    returns a list of all possible permutations of a input list of lists
    """
    tuple_list=list(itertools.product(*list_lists))
    permutation_list=[ list(x) for x in tuple_list]
    return(permutation_list)

def make_parameter_list_sensitivity():
    """
    Output: np array with a list of lists of parameter permutation for sensitivity
            analysis.
    """
    binarize_list=[5,20,40,80]
    eps_list=[1,2,5]
    min_samples=[1,2,5]
    filter_boundary_list=[50, 100, 200]
    remove_large_clusters_list=[0,1,10]
    remove_small_clusters_list=[1,50,800]
    max_filter_list=[1,3,5,7]
    eps_grain_boundary_list=[1,3]
    min_sample_grain_boundary_list=[1,3]
    binarize_bdr_coord_list=[80, 100, 120]
    binarize_grain_coord_list=[80, 100, 120]
    
    
    param_list= make_permutation_list(
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
    param_array=np.array(param_list)
    
    return (param_array)
# =============================================================================
# Sensitivity functions
# =============================================================================

def var_Y(y_vector):
    """
    Args: no array
    Output: variance as float
    """
    m=y_vector.mean()
    diff_v=0
    for i in range(len(y_vector)):
        diff_v+=(y_vector[i]-m)**2
    res= diff_v /len(y_vector)
    return(res)

def conditional_variance(my_func, X, k):
    """
    Fixes the value of the dimension k to all its possible values, 
    and calculates the mean of the resulting variance
    X np array of input list
    k is the dimension to be analyzed, i.e. the column k 
    f is the function to be analyzed, that receives the input
    list as an argument
    """
    dim_to_analyse=X[:,k]
    possible_values=set(dim_to_analyse) 
    variance_list=[]
    
    X_copy=np.copy(X)
    for value in possible_values:
        X_copy[:,k]=value
        y_res=[]
        
        for i in range(X.shape[0]):
            input_list=X_copy[i,:]
            y_res.append(my_func(input_list.reshape(1, -1)))
        
        variance_list.append(var_Y(np.array(y_res)))
    
    conditional_var_mean=np.mean(np.array(variance_list))
    
    
    return(conditional_var_mean)


def total_variance(my_func, X):
    """
    Calculates the total variances for given list of input lists
    X np array of input list
    f is the function to be analyzed, that receives the input
    list as an argument
    """
    y_res=[]
    for i in range(X.shape[0]):
        input_list=X[i,:]
        y_res.append(my_func(input_list.reshape(1, -1)))
         
    total_variance=var_Y(np.array(y_res))
    
    return(total_variance)


def s_first_order_k(my_func, X, k):
    """
    input a function my_func that receives a 1D numpy array as an input
    input a X array with the input values for the function in the rows
    k the dimension to be analyzed
    
    """
    conditional_var_mean=conditional_variance(my_func, X, k)
    tot_variance=total_variance(my_func, X)
    
    
    return(1-(conditional_var_mean/tot_variance))

def calc_first_order_s(my_func, X):
    """
    input a function my_func that receives a 1D numpy array as an input
    input a X array with the input values for the function in the rows
    output a list with the first order sensitivity coefficients
    """
    sensitivity_list=[]
    
    for i in range(X.shape[1]): 
       sensitivity_list.append(round(s_first_order_k(my_func, X, i),2)) 
    return (sensitivity_list)
    

