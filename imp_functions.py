import numpy as np

# sigmoid function for binary result 0<x<1
def sigmoid(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid

# tanh function for hidden layers -1<x<1
def tanh(x):
    tanh = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    return tanh

# ReLU function 
def ReLU(x):
    if x>0:
        rectifier = x
    else:
        rectifier = 0
    return rectifier
