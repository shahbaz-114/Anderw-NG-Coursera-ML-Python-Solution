import numpy as np
import pandas as pd
import warmUpExercise as WUE
import plotData as PD
import computeCost as CS
import gradientDescent as GD
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
#      gradientDescent.py
#      computeCost.py
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
# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = pd.read_csv('ex1data1.txt',header=None)
X,y  = data[0],data[1] #Data Separated into two pandas series (internally it is a numpy array (not matrix)
m    = np.size(y)      #number of training example
PD.plotData(X, y)
input("Press Enter to continue...")
# =================== Part 3: Cost and Gradient descent ===================
X     = np.stack([np.ones(m),X],axis=1) #Add X0 (x0 = 1) column to X
theta = np.zeros([2,1])                 #initialize fitting parameters
iterations = 1500                       #Gradient descent iteration limit
alpha      = 0.01                       #Greadient descent Alpha value (tweak this to see the change in learning rate
print('\nTesting the cost function ...\n')
J = CS.computeCost(X, y, theta)#compute and display initial cost
print('With theta= [0 ; 0]\nCost computed\t= ', J)
print('Expected cost value (approx) 32.07')

J = CS.computeCost(X, y, theta=np.array([[-1],[2]])) #further testing of the cost function
print('\nWith theta = [-1 ; 2]\nCost computed = ', J)
print('Expected cost value (approx) 54.24')
input("Press Enter to continue...")
print('\nRunning Gradient Descent ...\n')

theta = GD.gradientDescent(X, y, theta, alpha, iterations) #run gradient descent
print('Theta found by gradient descent:\n',theta)
print('Expected theta values (approx)')
print(' -3.6303\n  1.1664\n')

# Plot the linear fit
PD.plotData(X[:, 1], y)
plt.plot(X[:, 1], X@theta, '-')
plt.legend(['Training data', 'Linear regression'])
plt.draw()
plt.pause(0.001)

# Predict values for population sizes of 35,000 and 70,000
predict1 = (np.array([1, 3.5])@theta)[0]
predict2 = (np.array([1, 7])@theta)[0]
print('For population = 35,000, we predict a profit of', predict1*10000)
print('For population = 70,000, we predict a profit of', predict2*10000)
input("Press Enter to continue...")

#  ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')
#Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros([np.size(theta0_vals), np.size(theta1_vals)]) # initialize J_vals to a matrix of 0's
for i in range(np.size(theta0_vals)):
    for j in range(np.size(theta1_vals)):
        t = np.array([[theta0_vals[i]],[theta1_vals[j]]])
        J_vals[i,j] = CS.computeCost(X, y, t)

#surface plot
#Transpose the J_vals as per surface/countur plot requirement/ Else you will see filliped data in contours
J_vals = np.transpose(J_vals)
fig    = plt.figure()
ax     = plt.axes(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals,cmap='viridis')
ax.set_title('Cost Function Plot')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Surface')

fig1   = plt.figure()
plt.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
plt.title('Contour, Achived minima')
plt.show()