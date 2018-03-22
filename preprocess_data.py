import pandas as pd
import numpy as np
import os, csv

"""
Return a list of all the closing prices
"""
def get_closing_prices(directory, filename):
    path_to_data = os.path.join(directory, filename)
    raw_data = pd.read_csv(path_to_data)
    return raw_data['Close'].values.reshape(-1, 1)

"""
Take values from a list and write to .csv file
"""
def write_to_csv_file(directory, filename, data_list):
    path_to_file = os.path.join(directory, filename)
    out_file = open(path_to_file, 'w')
    for row in data_list:
        for col in row:
            out_file.write('%f' % col)
        out_file.write('\n')
    out_file.close()
    return path_to_file

"""
Normalizes the values within their respective windows: dividing by first price in window and subtracting 1
"""
def normalize(windows_data):
    windows = []
    for window in windows_data:
        result = [((float(price)/float(window[0])) - 1) for price in window]
        windows.append(result)
    return np.array(windows)

"""
Load in the dataset from .csv file and split into training/test set
"""
def split_dataset(filename, sequence_length, test_percent=0.1):
    # Get the Data from the .csv file
    read_file = open(filename, 'r').read()
    data = read_file.split('\n')

    # Split Dataset into windows of size sequence_length inputs plus size 1 output
    window_size = sequence_length + 1
    windows = []
    for index in range(len(data) - window_size):
        windows.append(data[index:index + window_size])

    # Normalize data (comes out as type np.array)
    windows = normalize(windows)

    end_of_training_set = int(windows.shape[0] * (1 - test_percent))

    # Split into training, test and inputs, outputs
    training_set = windows[:end_of_training_set, :]
    test_set = windows[end_of_training_set:, :]
    np.random.shuffle(training_set)
    x_train = training_set[:, :-1]
    y_train = training_set[:, -1]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Keep a copy of the real y-values for the test set (for visualizing data better after testing)
    unnormalized_y_test = data[end_of_training_set:end_of_training_set+len(y_test)]

    return x_train, y_train, x_test, y_test, unnormalized_y_test
