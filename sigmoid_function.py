import numpy as np
# define sigmoid function for prediction

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
