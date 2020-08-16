# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:28:03 2020

@author: hto_r
"""
# =============================================================================
#  Note I wrote this helper functions without knowing about itertools
#  know knowing that it exists, they are obsolete. In the next version they will
#   disappear 
# =============================================================================
    
def make_param_list_5(l1, l2, l3, l4, l5):
    """Function to make a list of all possible permutations of several list
    args: several lists li
    return: list of lists with all possible permutations
    """
    combination_list=[]
    for i in l1:
        for j in l2:
            for k in l3:
                for l in l4:
                    for m in l5:
                        combination_list.append((i,j,k,l,m))
    return(combination_list)
    
def make_param_list_6(l1, l2, l3, l4, l5, l6):
    """Function to make a list of all possible permutations of several list
    args: several lists li
    return: list of lists with all possible permutations
    """
    combination_list=[]
    for i in l1:
        for j in l2:
            for k in l3:
                for l in l4:
                    for m in l5:
                        for n in l6:
                            combination_list.append((i,j,k,l,m, n))
    return(combination_list)
    
def make_param_list_9(l1, l2, l3, l4, l5, l6, l7, l8, l9):
    """Function to make a list of all possible permutations of several list
    args: several lists li
    return: list of lists with all possible permutations
    """
    combination_list=[]
    for i in l1:
        for j in l2:
            for k in l3:
                for l in l4:
                    for m in l5:
                        for n in l6:
                            for o in l7:
                                for p in l8:
                                    for q in l9:
                                        combination_list.append((i,j,k,l,m,n,o,p,q))
    return(combination_list)    
    
def make_param_list_11(l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11):
    """Function to make a list of all possible permutations of several list
    args: several lists li
    return: list of lists with all possible permutations
    """
    combination_list=[]
    for i in l1:
        for j in l2:
            for k in l3:
                for l in l4:
                    for m in l5:
                        for n in l6:
                            for o in l7:
                                for p in l8:
                                    for q in l9:
                                        for r in l10:
                                            for s in l11:
                                                combination_list.append((i,j,k,l,m,n,o,p,q,r,s))
    return(combination_list)

def make_param_list_15(l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15):
    """Function to make a list of all possible permutations of several list
    args: several lists li
    return: list of lists with all possible permutations
    """
    combination_list=[]
    for i in l1:
        for j in l2:
            for k in l3:
                for l in l4:
                    for m in l5:
                        for n in l6:
                            for o in l7:
                                for p in l8:
                                    for q in l9:
                                        for r in l10:
                                            for s in l11:
                                                for t in l12:
                                                    for u in l13:
                                                        for v in l14:
                                                            for w in l15:
                                                                combination_list.append((i,j,k,l,m,n,o,p,q,r,s,t,u,v,w))
    return(combination_list)