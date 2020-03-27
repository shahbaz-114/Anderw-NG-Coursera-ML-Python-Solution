import numpy as np
def normalEqn(X, y):
    '''
    computes the closed-form solution to linear regression using the normal equations.
    '''
    theta  = ((np.linalg.pinv((np.transpose(X)@X)))@(np.transpose(X)))@y
    return theta

