import keras_impl_lstm as kil
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

"""
Evaluate different batch_sizes
"""
def tune_batch_size(batch_sizes):
    # Hold other hyperparameters constant
    results = "Constant Hyperparameters: num_epochs=10, timestep=50, cell_state_size=20, num_layers=1 \n"
    results += "Batch_size\tRMSE\n"
    for batch_size in batch_sizes:
        predictions, ground_truth = kil.lstm_predict(batch_size=batch_size)
        rmse = sqrt(mean_squared_error(ground_truth, predictions))
        results += str(batch_size) + '\t' + str(rmse) + "\n"

    # Write out the results to .txt file for evaluation
    batch_file = open('./hyperparameters/batch_size_results.txt', 'w')
    batch_file.write(results)
    batch_file.close()

"""
Evaluate different cell_state_sizes (hidden units in each layer)
"""
def tune_cell_state_size(cell_state_sizes, chosen_batch_size):
    # Hold other hyperparameters constant
    results = "Constant Hyperparameters: batch_size=" + str(chosen_batch_size) + ", num_epochs=10, timestep=50, num_layers=1 \n"
    results += "cell_state_size\tRMSE\n"
    for css in cell_state_sizes:
        predictions, ground_truth = kil.lstm_predict(batch_size=chosen_batch_size, cell_state_size=css)
        rmse = sqrt(mean_squared_error(ground_truth, predictions))
        results += str(css) + '\t' + str(rmse) + '\n'

    # Write out the results to .txt file for evaluation
    css_file = open('./hyperparameters/cell_state_size_results.txt', 'w')
    css_file.write(results)
    css_file.close()

"""
Evaluate timesteps (window sizes)
"""
def tune_timesteps(timesteps, chosen_batch_size, chosen_css):
    # Hold other hyperparameters constant
    results = "Constant Hyperparameters: batch_size=" + str(chosen_batch_size) + ", num_epochs=10, cell_state_size=" + str(chosen_css) + ", num_layers=1 \n"
    results += "timesteps\tRMSE\n"
    for timestep in timesteps:
        predictions, ground_truth = kil.lstm_predict(batch_size=chosen_batch_size, timestep=timestep, cell_state_size=chosen_css)
        rmse = sqrt(mean_squared_error(ground_truth, predictions))
        results += str(timestep) + '\t' + str(rmse) + "\n"

    # Write out the results to .txt file for evaluation
    timestep_file = open('./hyperparameters/timestep_results.txt', 'w')
    timestep_file.write(results)
    timestep_file.close()

"""
Evaluate number of layers
"""
def tune_num_layers(layers, chosen_batch_size, chosen_css, chosen_timestep):
    results = "Constant Hyperparameters: batch_size=" + str(chosen_batch_size) + ", num_epochs=10, timestep=" + str(chosen_timestep) + ", cell_state_size=" + str(chosen_css) + ", num_layers=1 \n"
    results += 'num_layers\tRMSE\n'
    for num_layers in layers:
        layer_sizes = [1]
        for _ in range(num_layers):
            layer_sizes.append(chosen_css)  # Whatever cell_state_size evaluated out to be
        layer_sizes.append(1)
        predictions, ground_truth = kil.lstm_predict(batch_size=chosen_batch_size, timestep=chosen_timestep, layer_sizes=layer_sizes)
        rmse = sqrt(mean_squared_error(ground_truth, predictions))
        results += str(layer_sizes) + '\t' + str(rmse) + "\n"

    # Write out the results to .txt file for evaluation
    layers_file = open('./hyperparameters/num_layers_results.txt', 'w')
    layers_file.write(results)
    layers_file.close()

"""
Evaluate number of epochs
"""
def tune_num_epochs(epochs, chosen_batch_size, chosen_css, chosen_timestep, chosen_num_layers):
    results = 'Constant Hyperparameters: batch_size=' + str(chosen_batch_size) + ', timestep=' + str(chosen_timestep) + ', cell_state_size=' + str(chosen_css) + ', layers=' + str(chosen_num_layers) + ' \n'
    results += 'num_epochs\tRMSE\n'
    layer_sizes = [1]
    for _ in range(chosen_num_layers):  # Whatever num layers should be
        layer_sizes.append(chosen_css)  # Whatever cell_state_size evaluated out to be
    layer_sizes.append(1)

    for num_epochs in epochs:
        predictions, ground_truth = kil.lstm_predict(batch_size=chosen_batch_size, num_epochs=num_epochs, timestep=chosen_timestep, layer_sizes=layer_sizes)
        rmse = sqrt(mean_squared_error(ground_truth, predictions))
        results += str(num_epochs) + '\t' + str(rmse) + "\n"

    # Write out the results to .txt file for evaluation
    epochs_file = open('./hyperparameters/num_epochs_results.txt', 'w')
    epochs_file.write(results)
    epochs_file.close()
