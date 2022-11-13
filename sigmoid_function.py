# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:10:26 2022

@author: pc
"""

import numpy as np
# define sigmoid function for prediction

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s