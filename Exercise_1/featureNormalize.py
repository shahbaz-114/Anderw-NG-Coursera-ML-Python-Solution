import numpy as np
import pandas as pd
def featureNormalize(X):
    '''
    FUNCTION TO RETURN THE MEAN AND SIGMA (VARIANCE) OF EACH FEATURE (x) and NORMALIZE THE DATA SET TO HAVE A ZERO MEAN AND ONE SIGMA
    '''
    mu     = X.mean()
    sigma  = X.std()
    X_norm = (X-mu)/sigma
    return X_norm,mu,sigma