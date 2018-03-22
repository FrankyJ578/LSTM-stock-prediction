from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy as np

"""
Build a Neural Net with only 1 LSTM layer
"""
def single_layer_LSTM(layer_sizes):
    model = Sequential()

    model.add(LSTM(layer_sizes[1], return_sequences=False, input_shape=(None, layer_sizes[0])))
    model.add(Dropout(0.2))

    model.add(Dense(units=layer_sizes[-1]))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    return model

"""
Build a Neural Net with multiple LSTM layers
"""
def multiple_layer_LSTM(layer_sizes):
    model = Sequential()

    model.add(LSTM(layer_sizes[1], return_sequences=True, input_shape=(None, layer_sizes[0])))
    model.add(Dropout(0.2))

    # Loop over the intermediate layers with return_sequences=True
    for l in range(2, len(layer_sizes)-2):
        model.add(LSTM(layer_sizes[l], return_sequences=True))
        model.add(Dropout(0.2))

    # Final LSTM Layer
    model.add(LSTM(units=layer_sizes[-2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=layer_sizes[-1]))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    return model
