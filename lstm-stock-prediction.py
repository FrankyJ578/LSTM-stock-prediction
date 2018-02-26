import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

from sklearn.preprocessing import StandardScaler

DATA_DIR = 'data'

# Load the Dataset (currently just the prices of SP500)
epochs = 200
num_steps = 10
batch_size = 7
path_to_data = os.path.join(DATA_DIR, 'AAPL.csv')
raw_data = pd.read_csv(path_to_data)
closing_prices = raw_data['Close'].values.reshape(-1, 1)
print 'Total number of days in the dataset: {}'.format(len(closing_prices))

# Scale the dataset so that it has mean 0, variance 1
scaler = StandardScaler()
scaled_closing_prices = scaler.fit_transform(closing_prices)

def split_into_windows(data, num_steps):
    X, Y = [], []

    for i in range(len(data)-num_steps):
        X.append(data[i:i+num_steps])
        Y.append(data[i+num_steps])
    return np.array(X), np.array(Y)

# Split the dataset into train/test
def train_test_split(X, Y, batch_size, test_ratio):
    num_train = (int(len(X) * (1.0-test_ratio)) // batch_size)*batch_size
    train_X, test_X = X[:num_train], X[num_train:]
    train_Y, test_Y = Y[:num_train], Y[num_train:]
    return train_X, train_Y, test_X, test_Y

# Preprocess the data
def preprocess_data(data, num_steps, batch_size, test_ratio=0.1):
    X, Y = split_into_windows(data, num_steps)
    return train_test_split(X, Y, batch_size, test_ratio)

train_X, train_Y, test_X, test_Y = preprocess_data(scaled_closing_prices, num_steps, batch_size)
print "X_train size: {}".format(train_X.shape)
print "y_train size: {}".format(train_Y.shape)
print "X_test size: {}".format(test_X.shape)
print "y_test size: {}".format(test_Y.shape)

# Create the RNN-LSTM

def createLSTMCell(layer_size, batch_size, num_layers, keep_prob):
    layer = tf.contrib.rnn.BasicLSTMCell(layer_size)

    if keep_prob < 1.0:
        layer = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=keep_prob)

    cell = tf.contrib.rnn.MultiRNNCell([layer]*num_layers)
    init = cell.zero_state(batch_size, tf.float32)

    return cell, init

def outputLayer(output, input_size, output_size):
    a = output[:,-1,:]
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.05), name='output_weights')
    b = tf.Variable(tf.zeros([output_size]), name='output_bias')
    return tf.matmul(a, W) + b

def minimizeLoss(predictions, targets, learning_rate):
    loss = tf.reduce_mean(tf.square(predictions-targets), name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return loss, optimizer

class LSTMStockPrediction():
    def __init__(self, learning_rate=0.001, batch_size=7, layer_size=512, num_layers=1, keep_prob=0.9, num_steps=10):
        self.inputs = tf.placeholder(tf.float32, [batch_size, num_steps, 1], name='input')
        self.targets = tf.placeholder(tf.float32, [batch_size, 1], name='target')

        cell, init = createLSTMCell(layer_size, batch_size, num_layers, keep_prob)
        outputs, states = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init)
        self.predictions = outputLayer(outputs, layer_size, 1)
        self.loss, self.optimizer = minimizeLoss(self.predictions, self.targets, learning_rate)


tf.reset_default_graph()
model = LSTMStockPrediction()

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(epochs):
    predictions, epoch_loss = [], []
    num_batches = int(len(train_X) // batch_size)
    if batch_size * num_batches < len(train_X):
        num_batches += 1

    batch_indices = range(num_batches)
    random.shuffle(batch_indices)
    for j in batch_indices:
        batch_X, batch_Y = None, None
        if j == len(batch_indices)-1:
            batch_X = train_X[j*batch_size:]
            batch_Y = train_Y[j*batch_size:]
        else:
            batch_X = train_X[j*batch_size:(j+1)*batch_size]
            batch_Y = train_Y[j*batch_size:(j+1)*batch_size]

        o, c, _ = session.run([model.predictions, model.loss, model.optimizer], feed_dict={model.inputs:batch_X, model.targets:batch_Y})

        epoch_loss.append(c)
        predictions.append(o)
    # trained_scores.append(predictions)
    if (i % 5) == 0:
        print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))


training_results = []
for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        training_results.append(predictions[i][j])

test = []
num_batches = int(len(test_X) // batch_size)
if batch_size * num_batches < len(test_X):
    num_batches += 1

batch_indices = range(num_batches)
for i in batch_indices:
    batch_X = None
    if i == len(batch_indices)-1:
        batch_X = test_X[i*batch_size:]
    else:
        batch_X = test_X[i*batch_size:(i+1)*batch_size]

    o = session.run([model.predictions], feed_dict={model.inputs:batch_X})
    test.append(o)

test_new = []
for i in range(len(test)):
    for j in range(len(test[i][0])):
        test_new.append(test[i][0][j])

test_results = []
for i in range(len(train_X)+len(test_X)):
    if i >= len(train_X)+1:
        test_results.append(test_new[i-(len(train_X)+1)])
    else:
        test_results.append(None)

plt.figure(figsize=(16, 7))
plt.plot(scaled_closing_prices, label='Original data')
plt.plot(training_results, label='Training data')
plt.plot(test_results, label='Testing data')
plt.legend()
plt.show()
