import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

def plot_results(predictions, ground_truth):
    plot = plt.figure()
    ax = plot.add_subplot(111)
    ax.plot(ground_truth, label='Ground-Truth')
    plt.plot(predictions, label='Prediction', color='red')
    ax.set_xlabel('Days')
    ax.set_ylabel('Closing Stock Prices')
    plt.show()

def plot_multiple_results(predictions, ground_truth, prediction_len):
    plot = plt.figure()
    ax = plot.add_subplot(111)
    ax.plot(ground_truth, label='Ground-Truth')
    ax.set_xlabel('Days')
    ax.set_ylabel('Closing Stock Prices')
    for i, prediction in enumerate(predictions):
        plt.plot(range(i*prediction_len, (i+1)*prediction_len), prediction, label='Prediction', color='red')
    plt.show()
