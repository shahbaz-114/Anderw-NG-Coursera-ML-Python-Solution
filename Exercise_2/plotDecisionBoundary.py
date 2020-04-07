import matplotlib.pyplot as plt
import numpy as np
import plotData as PD
import mapFeature as MP


def plotDecisionBoundary(theta, X, y):
    PD.plotData(X[:,[1,2]], y)
    if X.shape[1] <= 3:
        slope     = -theta[1]/theta[2]
        intercept = -theta[0]/theta[2]
        plot_x    = np.array([min(X[:,1])-2, max(X[:,2])+2])  #X-co-ordinate of the end points of decision line
        plot_y    = slope*plot_x + intercept #Decison line
        plt.plot(plot_x, plot_y, c="orange", label="decision boundary")
        plt.legend()
    else:
        u_vals = np.linspace(-1, 1.5, 50)
        v_vals = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u_vals), len(v_vals)))
        for i in range(len(u_vals)):
            for j in range(len(v_vals)):
                X1 = np.array([u_vals[i]])
                X2 = np.array([v_vals[j]])
                z[i, j] = MP.mapFeature(X1,X2)@theta
        plt.contour(u_vals, v_vals, z.T, 0)
