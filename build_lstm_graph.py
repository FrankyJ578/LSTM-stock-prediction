import tensorflow as tf
import numpy as np
import random

from lstm_params import PARAMS

def createLstmCell(params):
    lstm_cell = tf.contrib.rnn.LSTMCell(params.layer_size, state_is_tuple=True)
    # Check/Perform Dropout
    if params.keep_prob < 1.0:
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=params.keep_prob)
    return lstm_cell

def buildLstmGraph(params=None):
    tf.reset_default_graph()    # Just to make sure no junk remaining
    lstmGraph = tf.Graph()

    # Check if we want default parameters or if provided parameters
    if params is None:
        params = PARAMS

    # Set placeholders for the inputs, ground_truths, and learning_rate (values inputted later)
    inputs = tf.placeholder(tf.float32, [None, params.num_steps, params.window_size], name='inputs')
    truths = tf.placehold(tf.float32, [None, params.window_size], name='truths')
    learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')

    # create Lstm cell
    cell = None
    if params.num_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell([createLstmCell(params) for _ in range(params.num_layers)], state_is_tuple=True)
    else:
        cell = createLstmCell(params)

    # create RNN with cell
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    outputs = tf.transpose(outputs, [1, 0, 2])  # make the shape (num_steps, batch_size, layer_size)

    # output layer
    last_output = tf.gather(val, int(val.get_shape()[0]) - 1, name='last_output')
    W = tf.get_variable("W", [params.layer_size, params.input_size], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    bias = tf.get_variable("bias", [params.input_size], initializer=tf.zeros_initializer())
    prediction = tf.matmul(last_output, W) + bias

    # train
    cost = tf.reduce_mean(tf.square(prediction - truths), name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    return lstmGraph
