from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


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


# Generate data to train
X, y = make_blobs(n_samples=1000, centers=2)

# Show the generated data
plot_data_xy(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Create model
model = Sequential()
model.add(Dense(1, input_shape=(2,), activation="sigmoid"))
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=1)
evaluate_result = model.evaluate(X_test, y_test)
print(f"loss: {evaluate_result[0]} \naccuracy: {evaluate_result[1]}")
plot_decision_boundary(model, X, y)
