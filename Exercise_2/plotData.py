import matplotlib.pyplot as plt


def plotData(X, y):
    pos_x0 = X[(y == 1),0]
    pos_x1 = X[(y == 1),1]
    neg_x0 = X[(y == 0),0]
    neg_x1 = X[(y == 0),1]
    plt.plot(pos_x0, pos_x1, 'k+', mfc='k', ms=8,label="Admitted")
    plt.plot(neg_x0, neg_x1, 'ko', mfc='y', ms=8,label="Not-Admitted")





