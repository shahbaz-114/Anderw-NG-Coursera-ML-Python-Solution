import numpy as np

def computeCost(X, y, theta):
    '''
    GENERATE THE COST FOR A GIVEN THETA VALUE (theta is the vector contaion all theta's (theta0, theta1..etc.)
    '''
    m = np.size(y)                   # number of training examples
    y = np.array(y).reshape(m,1)     #convert into a 1D vector -- Y was a simple array and not a matrix
    J = ((np.transpose(X@theta -y))@(X@theta -y))/(2*m)
    return J[0,0]                    #J will be in 1x1 matrix, we will return the scalar value



