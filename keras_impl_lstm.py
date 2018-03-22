from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import build_lstm, matplotlib, preprocess_data, prediction_methods, plotting_methods
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = '.'
DATA_DIR = 'data'
STOCK_CSV = 'SP500.csv'
STOCK_CLOSING_CSV = 'SP500-closing.csv'

def lstm_predict(batch_size=512, num_epochs=10, timestep=50, cell_state_size=20, layer_sizes=[1], prediction_length=1, predict_next=True, predict_tendency=False, predict_all=False, show_plot=False):

    def predict_next_day(model, x_test, y_test, unnormalized_y_test, show_plot):
        predictions = prediction_methods.predict_one_ahead(model, x_test)
        for i in range(len(y_test)):
        	y_test[i] = (y_test[i]+1) * float(unnormalized_y_test[i])
        	predictions[i] = (predictions[i]+1) * float(unnormalized_y_test[i])

        if show_plot: plotting_methods.plot_results(predictions, unnormalized_y_test)
        return predictions

    def predict_whole_sequence(model, x_test, unnormalized_y_test, show_plot):
        predictions = prediction_methods.predict_entire_sequence(model, x_test, timestep)
        for i in range(len(predictions)):
        	predictions[i] = (predictions[i]+1) * float(unnormalized_y_test[i])

        if show_plot: plotting_methods.plot_results(predictions, unnormalized_y_test)
        return predictions

    def predict_sequences(model, x_test, unnormalized_y_test, y_test, show_plot):
        predictions = prediction_methods.predict_in_sequences(model, x_test, window_size=timestep, prediction_length=50)
        col = len(predictions[0])
        tmp = np.asarray(predictions).reshape(1,-1)
        for i in range(len(tmp)):
        	tmp[i] = (tmp[i]+1) * float(unnormalized_y_test[i])
        predictions = tmp.reshape(-1,col).tolist()

        if show_plot: plotting_methods.plot_multiple_results(predictions, unnormalized_y_test, prediction_len=50)
        return predictions

    # Preprocess the data
    closing_prices = preprocess_data.get_closing_prices(DATA_DIR, STOCK_CSV)
    path_to_file = preprocess_data.write_to_csv_file(CURRENT_DIR, STOCK_CLOSING_CSV, closing_prices)

    x_train, y_train, x_test, y_test, unnormalized_y_test = preprocess_data.split_dataset(path_to_file, timestep)

    # Build the model (defaults to single layer if more layer sizes are not provided)
    if len(layer_sizes) == 3: model = build_lstm.single_layer_LSTM(layer_sizes)
    elif len(layer_sizes) > 3: model = build_lstm.multiple_layer_LSTM(layer_sizes)
    else: model = build_lstm.single_layer_LSTM([1, cell_state_size, 1])

    # Fit model to training set
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.05)

    # Choose Prediction Type
    if predict_next:
        predictions = predict_next_day(model, x_test, y_test, unnormalized_y_test, show_plot)
    if predict_all:
        all_predictions = predict_whole_sequence(model, x_test, unnormalized_y_test, show_plot)
    if predict_tendency:
        tendency_predictions = predict_sequences(model, x_test, unnormalized_y_test, y_test, show_plot)

    if predict_next:
        return predictions, y_test
