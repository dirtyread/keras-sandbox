# genial sample generator - example of circle distribution
from sklearn.datasets import make_circles
# generic split data for train and test
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# my utility function
import my_utils_function as muf
# for visualization
from keras.utils import plot_model
# to callback function - early stopping when train
from keras.callbacks import EarlyStopping


X, y = make_circles(n_samples=1000, factor=.6, noise=.1, random_state=42)
muf.plot_data_xy(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# I create two hidden layer for circles data representation.
# Two hidden layrer can represent arbitrary decision boundary
model = Sequential()
model.add(Dense(4, input_shape=(2,), activation="tanh", name="hidden_1"))
model.add(Dense(4, activation="tanh", name="hidden_2"))
model.add(Dense(1, activation="sigmoid", name="output_layer"))
# visualization
model.summary()
model.compile(Adam(lr=.05), 'binary_crossentropy', metrics=['accuracy'])
# visualization
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
my_callbacks = [EarlyStopping(monitor="val_acc", patience=5, mode='max')]
model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=my_callbacks, validation_data=(X_test, y_test))

evaluate_result = model.evaluate(X_test, y_test)
print(f"loss:{evaluate_result[0]}\naccuracy:{evaluate_result[1]}")
muf.plot_decision_boundary(model, X, y)
