import numpy as np
# function that create zero matrices

def zeros(dim):
    # Weight matrix
    w = np.zeros((dim, 1))
    # Bias vector
    b = 0.0
    return w, b
