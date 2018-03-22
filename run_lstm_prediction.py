import keras_impl_lstm as kil
from chosen_hyperparameters import DEFAULT_CONFIG

def main(config=DEFAULT_CONFIG):
    layer_sizes = [1]
    for _ in range(config.num_layers):  # Whatever num layers should be
        layer_sizes.append(config.cell_state_size)  # Whatever cell_state_size evaluated out to be
    layer_sizes.append(1)

    # kil.lstm_predict(config.batch_size, config.num_epochs, config.timestep, config.cell_state_size, layer_sizes, show_plot=True)
    # kil.lstm_predict(config.batch_size, config.num_epochs, config.timestep, config.cell_state_size, layer_sizes, predict_next=False, predict_all=True, show_plot=True)
    kil.lstm_predict(config.batch_size, config.num_epochs, config.timestep, config.cell_state_size, layer_sizes, prediction_length=50, predict_next=False, predict_tendency=True, show_plot=True)


if __name__ == '__main__':
    main()
