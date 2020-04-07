import numpy as np
import sigmoid as SG
import math


def costFunction(theta, X, y):
    # %COSTFUNCTION Compute cost and gradient for logistic regression
    # %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    # %   parameter for logistic regression and the gradient of the cost
    # %   w.r.t. to the parameters.
    #
    # % Initialize some useful values
    m     = y.size # number of training examples
    y     = y.reshape(m,1)
    theta = theta.reshape(3,1)
    J     = 0

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Compute the cost of a particular choice of theta.
    # %               You should set J to the cost.
    # %               Compute the partial derivatives and set grad to the partial
    # %               derivatives of the cost w.r.t. each parameter in theta
    # %
    # % Note: grad should have the same dimensions as theta

    H = SG.sigmoid(X@theta)
    J = ((1/m)*((-y.transpose()@np.log(H)) - (1-y).transpose()@(np.log(1-H))))[0,0]
    grad = ((1/m)*(X.transpose())@(H-y)).flatten()
    return J,grad

