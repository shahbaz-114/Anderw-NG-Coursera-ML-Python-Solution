import numpy as np
import computeCost as CS
def gradientDescentMulti(X, y, theta, alpha, num_iters):
  '''
  GRADIENTDESCENT Performs gradient descent to learn theta, updates theta by taking num_iters gradient steps with learning rate alpha
  '''
  m = len(y)  # number of training examples
  y = np.array(y).reshape(m, 1)
  J_history = np.zeros([num_iters, 1])
  for i in range(num_iters):
    theta = theta - alpha * (1 / m) * np.transpose(X) @ (X @ theta - y)
    J_history[i] = CS.computeCost(X, y, theta)
  return theta,J_history
