from libraries import utilities, visualization
from autoencoder.libraries import utilities as utilities_autoencoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import *


from keras.layers import Input, LSTM, RepeatVector

from sklearn.metrics import mean_squared_error
from numpy import fft

np.random.seed(42)
pd.set_option('display.max_columns', 32)
np.set_printoptions(precision=5, suppress=True)


#########################################################
currency = 'GBPUSD'
df, df_returns = utilities_autoencoder.get_data()
columns = utilities_autoencoder.get_columns(df_returns, include_timestamp=False)

scaler = StandardScaler()
df_returns[columns] = scaler.fit_transform(df_returns[columns])
df_train = df_returns.iloc[:300000, ]
df_test = df_returns.iloc[300000:, ]
x_train = df_train[columns].values
x_test = df_test[columns].values


#from keras.datasets import mnist
#(x_train, _), (x_test, _) = mnist.load_data()
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#print(x_train.shape)
#print(x_test.shape)

############### NN
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l1, l2

input_dim = 32
encoded_dim = 12
activation = 'tanh'
regularizer = regularizers.l2(1)
optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
loss = 'mean_squared_error'

inputs = Input(shape=(input_dim,))
encoded_inputs = Input(shape=(encoded_dim,))

encoded = Dense(encoded_dim, activation=activation, activity_regularizer=regularizer)(inputs)
decoded = Dense(input_dim, activation=activation)(encoded)

encoder = Model(inputs=inputs, outputs=encoded)
autoencoder = Model(inputs=inputs, outputs=decoded)
decoder = Model(inputs=encoded_inputs, outputs=autoencoder.layers[-1](encoded_inputs))

autoencoder.compile(optimizer=optimizer, loss=loss)
autoencoder.fit(x_train, x_train, epochs=100, batch_size=100, shuffle=False, validation_data=(x_test, x_test))

df_train_encoded = encoder.predict(x_train)
df_test_encoded = encoder.predict(x_test)

df_train_decoded = df_train.copy()
df_test_decoded = df_test.copy()
df_train_decoded[columns] = decoder.predict(df_train_encoded)
df_test_decoded[columns] = decoder.predict(df_test_encoded)

result = utilities_autoencoder.get_errors(df_train, df_train_decoded, df_test, df_test_decoded, columns)
result
utilities_autoencoder.plot_decoded_data(df_train, df_train_decoded, df_test, df_test_decoded, columns, encoding_dim, 'CHFUSD_close')

############### LTSM
input_dim = 32
encoding_dim = 8
activation = 'tanh'
timesteps = 100

input_ = Input(shape=(timesteps, input_dim))
encoded = LSTM(encoding_dim)(input_)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(dim, return_sequences=True)(decoded)

autoencoder = Model(input_, decoded)
encoder = Model(input_, encoded)

autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')
autoencoder.fit(x_train, x_train, epochs=25, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

df_train_encoded = encoder.predict(x_train)
df_test_encoded = encoder.predict(x_test)

df_train_decoded = df_train.copy()
df_test_decoded = df_test.copy()
df_train_decoded[columns] = decoder.predict(df_train_encoded)
df_test_decoded[columns] = decoder.predict(df_test_encoded)

result = utilities_autoencoder.get_errors(df_train, df_train_decoded, df_test, df_test_decoded, columns)
utilities_autoencoder.plot_decoded_data(df_train, df_train_decoded, df_test, df_test_decoded, columns, encoding_dim, 'CHFUSD_close')

## LSTM

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

batch_size = 8

model = Sequential()
model.add(LSTM(8, return_sequences=True, stateful=True, input_shape=(batch_size, timesteps, input_dim)))
model.add(Dropout(0.5))
model.add(Dense(input_dim, activation='tanh'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mean_squared_error'])

model.fit(x_train, x_train, batch_size=256, epochs=1)
score = model.evaluate(x_test, x_test, batch_size=16)