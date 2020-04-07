import numpy as np
import sigmoid as SG


def costFunctionReg(theta, X, y, Lambda):
    # %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    # %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    # %   theta as the parameter for regularized logistic regression and the
    # %   gradient of the cost w.r.t. to the parameters.
    #
    # % Initialize some useful values
    m     = y.size
    y     = y.reshape(m,1)
    theta = theta.reshape(np.size(theta),1)
    grad  = np.zeros(theta.shape[0])
    H     = SG.sigmoid(X@theta)

    J          = ((1/m)*((-y.transpose()@np.log(H)) - (1-y).transpose()@(np.log(1-H))))[0,0] +(Lambda/(2*m))*((theta[1:,0]**2).sum())
    grad_temp1 = ((1/m)*(X.transpose())@(H-y)).flatten()
    grad_temp2 = (((1/m)*(X.transpose())@(H-y)) + (Lambda/m)*theta).flatten()
    grad[0]  = grad_temp1[0]
    grad[1:] = grad_temp2[1:]
    return J,grad
