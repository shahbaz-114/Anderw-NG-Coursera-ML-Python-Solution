import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import costFunction as CF
import plotData as PD
import sigmoid as SG
import scipy.optimize as OP
import plotDecisionBoundary as PDB
import predict as PR


#  Machine Learning Online Class - Exercise 2: Logistic Regression
#
#   Instructions
#   ------------
#
#   This file contains code that helps you get started on the logistic
#   regression exercise. You will need to complete the following functions
#   in this exericse:
#
#      sigmoid.m
#      costFunction.m
#      predict.m
#      costFunctionReg.m
#
#   For this exercise, you will not need to change any code in this file,
#   or any other files other than those mentioned above.
#
#
# % Initialization
# clear ; close all; clc
#
# % Load Data
#   The first two columns contains the exam scores and the third column
#   contains the label.

data = pd.read_csv('ex2data1.txt',header=None)
X,y  = data.iloc[:,:2].to_numpy(),data.iloc[:,2].to_numpy()

# %% ==================== Part 1: Plotting ====================
#   We start the exercise by first plotting the data to understand the
#   the problem we are working with.
#
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
#
PD.plotData(X,y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.draw()
plt.pause(0.001)
#
# % Put some labels
# hold on;
# % Labels and Legend

#
input("Press Enter to continue...")
#
#
# % ============ Part 2: Compute Cost and Gradient ============
#   In this part of the exercise, you will implement the cost and gradient
#   for logistic regression. You neeed to complete the code in
#   costFunction.m
#
#   Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = X.shape
#
# % Add intercept term to x and X_test
X = np.concatenate([np.ones((m, 1)), np.array(X)], axis=1)
#
# % Initialize fitting parameters
initial_theta = np.zeros(n + 1)

# % Compute and display initial cost and gradient
[cost, grad] = CF.costFunction(initial_theta, X, y)
print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')
#
# % Compute and display cost and gradient with non-zero theta
test_theta   = np.array([-24,0.2,0.2])
[cost, grad] = CF.costFunction(test_theta, X, y)
#
print('Cost at test theta:', cost)
print('Expected cost (approx): 0.218')
print('Gradient at test theta: \n')
print(grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')
input("Press Enter to continue...")
#
# %% ============= Part 3: Optimizing using fminunc  =============
# %  In this exercise, you will use a built-in function (fminunc) to find the
# %  optimal parameters theta.
#
options= {'maxiter': 400}

# See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
opt_result = OP.minimize(CF.costFunction,
                        initial_theta,
                        (X, y),
                        jac=True,
                        method='TNC',
                        options=options)
cost, theta = opt_result.fun, opt_result.x
print('Cost at theta found by fminunc:', cost)
print('Expected cost (approx): 0.203\n')
print('theta:', theta)
print('Expected theta (approx):\n-25.161\n 0.206\n 0.201\n')
#
#  Plot Boundary
PDB.plotDecisionBoundary(theta, X, y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.draw()
plt.pause(0.001)
input("Press Enter to continue...\n")


# %% ============== Part 4: Predict and Accuracies ==============
# %  After learning the parameters, you'll like to use it to predict the outcomes
# %  on unseen data. In this part, you will use the logistic regression model
# %  to predict the probability that a student with score 45 on exam 1 and
# %  score 85 on exam 2 will be admitted.
# %
# %  Furthermore, you will compute the training and test set accuracies of
# %  our model.
# %
# %  Your task is to complete the code in predict.m
#
# %  Predict probability for a student with score 45 on exam 1
# %  and score 85 on exam 2
#
prob = SG.sigmoid(np.array([1,45,85]).reshape(1,3)@theta.reshape(3,1))[0,0]
print('For a student with scores 45 and 85, we predict an admission probability of:', prob)
print('Expected value: 0.775 +/- 0.002')
#
# % Compute accuracy on our training set
p = PR.predict(theta, X)
train_accuracy = np.mean(p == y)*100
print('Train Accuracy:',train_accuracy)
print('Expected accuracy (approx): 89.0\n')

