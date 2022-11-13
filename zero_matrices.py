# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:16:19 2022

@author: pc
"""
import numpy as np
# function that create zero matrices

def zeros(dim):
    # Weight matrix
    w = np.zeros((dim, 1))
    # Bias vector
    b = 0.0
    return w, b