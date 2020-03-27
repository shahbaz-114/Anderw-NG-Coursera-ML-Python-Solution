import numpy as np
import pandas as pd
import warmUpExercise as WUE
import featureNormalize as FN
import normalEqn as NE
import gradientDescentMulti as GDM
import matplotlib.pyplot as plt

# Machine Learning Online Class - Exercise 1: Linear Regression
#
#   Instructions
#   ------------
#
#   This file contains code that helps you get started on the
#   linear exercise. You will need to complete the following functions
#   in this exericse:
#
#      warmUpExercise.py
#      plotData.py
#      gradientDescent.py
#      computeCost.py
#      gradientDescentMulti.py
#      computeCostMulti.py
#      featureNormalize.py
#      normalEqn.py
#
#   For this exercise, you will not need to change any code in this file,
#   or any other files other than those mentioned above.
#
#  x refers to the population size in 10,000s
#  y refers to the profit in $10,000s
#
#
# Initialization
#
# ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
print(WUE.warmUpExercise())
input("Press Enter to continue...")
data = pd.read_csv('ex1data2.txt',header=None)
X,y  = data.iloc[:,:2],data.iloc[:,2] #Data Separated into two pandas series
m    = np.size(y)      #number of training example

#Print out some data points
print('First 10 examples from the dataset: \n')
print('Printing X\n',X.head(10))
print('Printing y\n',y.head(10))

# Scale features and set them to zero mean
print('Normalizing Features ...\n')
[X,mu,sigma] = FN.featureNormalize(X)
# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), np.array(X)], axis=1)
input("Press Enter to continue...")

# ================ Part 2: Gradient Descent ================
print('Running gradient descent ...\n')
alpha     = 0.01
num_iters = 400
theta = np.zeros([3, 1])
[theta, J_history] = GDM.gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(np.arange(0,num_iters),J_history,"b-")
plt.xlabel('Number of Iteration')
plt.ylabel('Cost J')
plt.title('Gradient Descent')
plt.draw()
plt.pause(0.001)

#Display gradient descent's result
print('Theta computed from gradient descent: \n',theta)

# % Estimate the price of a 1650 sq-ft, 3 br house
# % Recall that the first column of X is all-ones. Thus, it does not need to be normalized.
price = ((np.array([[1,(1650-mu[0])/sigma[0],(3-mu[1])/sigma[1]]]))@theta)[0,0]
print("Estimated price of a 1650 sq-ft, 3 br house $",price)
input("Press Enter to continue...")

# ================ Part 3: Normal Equations ================
print('Solving with normal equations...\n')
data  = pd.read_csv('ex1data2.txt',header=None)
X,y   = data.iloc[:,:2],data.iloc[:,2] #Data Separated into two pandas series
X     = np.concatenate([np.ones((m, 1)), np.array(X)], axis=1)
theta = NE.normalEqn(X, y)

print('Theta computed from the normal equations\n',theta)
price = np.array([[1,1650,3]])@theta
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n', price)

