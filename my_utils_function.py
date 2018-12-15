import matplotlib.pyplot as plt
import numpy as np


def plot_data_xy(X, y):
    """
    Plot data
    :param X: X dataset
    :param y: y dataset
    """
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'sb', alpha=0.5)
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'xr', alpha=0.5)
    plt.legend(['0', '1'])
    plt.show()


def plot_decision_boundary(model, X, y):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    c = model.predict(ab)
    Z = c.reshape(aa.shape)
    plt.figure(figsize=(12, 8))
    plt.contourf(aa, bb, Z, cmap='bwr', alpha=0.2)
    plot_data_xy(X, y)
