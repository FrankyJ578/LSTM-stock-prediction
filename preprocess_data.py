import numpy as np
import os
import pandas as pd
import random
import time

random.seed(time.time())

class StockDataSet(object):
    def __init__(self,
                 stock_sym,
                 window_size=1,
                 num_steps=30,
                 test_ratio=0.1,
                 normalized=True,
                 close_price_only=True):
        self.stock_sym = stock_sym
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.close_price_only = close_price_only
        self.normalized = normalized

        # Read csv file
        raw_data = pd.read_csv(os.path.join("data", "%s.csv" % stock_sym))

        # Merge into one sequence
        if close_price_only:
            self.raw_prices = raw_data['Close'].tolist()
        else:
            self.raw_prices = [price for tup in raw_data[['Open', 'Close']].values for price in tup]

        self.raw_prices = np.array(self.raw_prices)
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.prepare_data(self.raw_prices)

    def info(self):
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_sym, len(self.train_X), len(self.test_Y))

    def prepare_data(self, seq):
        # split into non-overlapping windows
        seq = [np.array(seq[i * self.window_size: (i + 1) * self.window_size]) for i in range(len(seq) // self.window_size)]

        # Normalize by taking the last price from t-1 window and divide by that across all values in window t
        if self.normalized:
            seq = [seq[0] / seq[0][0] - 1.0] + [curr / seq[i-1][-1] - 1.0 for i, curr in enumerate(seq[1:])]

        # split into groups of num_steps
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        Y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

        num_training_examples = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:num_training_examples], X[num_training_examples:]
        train_Y, test_Y = Y[:num_training_examples], Y[num_training_examples:]
        return train_X, train_Y, test_X, test_Y

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = range(num_batches)
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y
