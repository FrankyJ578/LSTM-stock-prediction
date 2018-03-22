from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy as np

"""
Predict the next day closing price
"""
def predict_one_ahead(model, data):
    prediction = model.predict(data)
    prediction = prediction.reshape((prediction.size,))
    return prediction

"""
Predict entire sequence by starting with first window, making a prediction,
shifting the window over by 1, and appending that prediction to the end of next
window.
"""
def predict_entire_sequence(model, data, window_size):
    # Start with the first window
    curr_window = data[0]
    predictions = []
    for _ in range(len(data)):
        # Predict the next day
        predictions.append(model.predict(curr_window[np.newaxis,:,:])[0,0])
        # Shift window over by 1 (with the next day prediction appended)
        curr_window = curr_window[1:]
        curr_window = np.insert(curr_window, [window_size-1], predictions[-1], axis=0)

    return predictions

"""
Predict sequences by same process as predict_entire_sequence, except only for
prediction_length days. After prediction_length days, use the test set window and
repeat. This is mainly for tendency predictions (generally trending up, or generally
trending down).
"""
def predict_in_sequences(model, data, window_size, prediction_length):
    prediction_sequences = []
    # Partition the test set into groups of prediction_len size
    for i in range(len(data)/prediction_length):
        curr_window = data[i*prediction_length]    # Starting window
        predictions = []
        for _ in range(prediction_length):
            predictions.append(model.predict(curr_window[np.newaxis,:,:])[0,0])  # Next day prediction
            # Shift window over by 1 (with next day prediction appended)
            curr_window = curr_window[1:]
            curr_window = np.insert(curr_window, [window_size-1], predictions[-1], axis=0)
        prediction_sequences.append(predictions)

    return prediction_sequences
