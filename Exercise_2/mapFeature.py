import numpy as np


def mapFeature(X1, X2):
    degree = 6
    out = np.ones((np.shape(X1)[0],1))
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            feature = ((X1**(i - j))*(X2**j)).reshape(np.shape(X1)[0],1)
            out     = np.hstack((out,feature))
    return out
