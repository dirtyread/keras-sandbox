# genial sample generator - example of circle distribution
from sklearn.datasets import make_circles
# generic split data for train and test
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input
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

input_layer = Input(shape=(2,))
x = Dense(4, activation="tanh", name="hidden-1")(input_layer)
x = Dense(4, activation="tanh", name="hidden-2")(x)
output_layer = Dense(1, activation="sigmoid", name="output_layer")(x)
# create model based on above definiction
model = Model(inputs=input_layer, outputs=output_layer)

model.summary()
model.compile(Adam(lr=.05), 'binary_crossentropy', metrics=['accuracy'])
# visualization
plot_model(model, to_file="model-api.png", show_shapes=True, show_layer_names=True)
# callback
my_callbacks = [EarlyStopping(monitor="val_acc", patience=5, mode='max')]
model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=my_callbacks, validation_data=(X_test, y_test))

evaluate_result = model.evaluate(X_test, y_test)
print(f"loss:{evaluate_result[0]}\naccuracy:{evaluate_result[1]}")
muf.plot_decision_boundary(model, X, y)
