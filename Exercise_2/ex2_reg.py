import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotData as PD
import mapFeature as MP
import costFunctionReg as CFR
import scipy.optimize as OP
import plotDecisionBoundary as PDB
import predict as PR
# %% Machine Learning Online Class - Exercise 2: Logistic Regression
# %
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the second part
# %  of the exercise which covers regularization with logistic regression.
# %
# %  You will need to complete the following functions in this exericse:
# %
# %     sigmoid.m
# %     costFunction.m
# %     predict.m
# %     costFunctionReg.m
# %
# %  For this exercise, you will not need to change any code in this file,
# %  or any other files other than those mentioned above.
# %
#
# %% Initialization
# clear ; close all; clc
#
# %% Load Data
# %  The first two columns contains the X values and the third column
# %  contains the label (y).
data = pd.read_csv('ex2data2.txt',header=None)
X,y  = data.iloc[:,:2].to_numpy(),data.iloc[:,2].to_numpy()
PD.plotData(X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()
plt.draw()
plt.pause(0.001)
input("Press Enter to continue...")

# %% =========== Part 1: Regularized Logistic Regression ============
# %  In this part, you are given a dataset with data points that are not
# %  linearly separable. However, you would still like to use logistic
# %  regression to classify the data points.
# %
# %  To do so, you introduce more features to use -- in particular, you add
# %  polynomial features to our data matrix (similar to polynomial
# %  regression).
# %
#
# % Add Polynomial Features
#
# % Note that mapFeature also adds a column of ones for us, so the intercept
# % term is handled
X = MP.mapFeature(X[:,0], X[:,1])
#
# % Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])
#
# % Set regularization parameter lambda to 1
Lambda = 1
#
# % Compute and display initial cost and gradient for regularized logistic
# % regression
[cost, grad] = CFR.costFunctionReg(initial_theta, X, y, Lambda)

print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:', grad[:5])
print('Expected gradients (approx) - first five values only:')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')
input("Press Enter to continue...")

# % Compute and display cost and gradient
# % with all-ones theta and lambda = 10
test_theta   = np.ones([X.shape[1],1])
[cost, grad] = CFR.costFunctionReg(test_theta, X, y, 10)

print('\nCost at test theta (with lambda = 10):', cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:',grad[:5])
print('Expected gradients (approx) - first five values only:\n 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')
input("Press Enter to continue...")

# %% ============= Part 2: Regularization and Accuracies =============
# %  Optional Exercise:
# %  In this part, you will get to try different values of lambda and
# %  see how regularization affects the decision coundart
# %
# %  Try the following values of lambda (0, 1, 10, 100).
# %
# %  How does the decision boundary change when you vary lambda? How does
# %  the training set accuracy vary?
# %
#
# % Initialize fitting parameters
initial_theta = np.zeros([X.shape[1],1])
#
# % Set regularization parameter lambda to 1 (you should vary this)
Lambda = 1
options= {'maxiter': 400}

# See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
opt_result = OP.minimize(CFR.costFunctionReg,
                        initial_theta,
                        (X, y, Lambda),
                        jac=True,
                        method='TNC',
                        options=options)
cost, theta = opt_result.fun, opt_result.x

# % Plot Boundary
print("Plotting decision boundary")
PDB.plotDecisionBoundary(theta, X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.draw()
plt.pause(0.001)
input("Press Enter to continue...\n")

p = PR.predict(theta, X)
train_accuracy = np.mean(p == y)*100
print('Train Accuracy:',train_accuracy)
print('Expected accuracy (approx): 83.1.0\n')
plt.show()

