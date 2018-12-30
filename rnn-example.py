from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping

from keras.datasets import imdb

# skip_top - we don't skip any word
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000, skip_top=0)

x_train = sequence.pad_sequences(x_train, maxlen=400)
x_test = sequence.pad_sequences(x_test, maxlen=400)

# MODEL
model = Sequential()
model.add(Embedding(5000, 64))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# TRAIN
cbk_early_stopping = EarlyStopping(monitor='val_acc', patience=2, mode='max')
model.fit(x_train, y_train, 20, epochs=2, validation_data=(x_test, y_test), callbacks=[cbk_early_stopping])

scores, acc = model.evaluate(x_test, y_test, batch_size=20)
print('scores:', scores, ' acc:', acc)
