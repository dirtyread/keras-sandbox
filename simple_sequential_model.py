from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# utils function
import my_utils_function as muf


# Generate data to train
X, y = make_blobs(n_samples=1000, centers=2)

# Show the generated data
muf.plot_data_xy(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Create model
model = Sequential()
model.add(Dense(1, input_shape=(2,), activation="sigmoid"))
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=1)
evaluate_result = model.evaluate(X_test, y_test)
print(f"loss: {evaluate_result[0]} \naccuracy: {evaluate_result[1]}")
muf.plot_decision_boundary(model, X, y)
