import hyperparameter_tuning as hpt
from chosen_hyperparameters import DEFAULT_CONFIG

# The Different Hyperparameter Values to Test
BATCH_SIZES = [64, 128, 256, 512]
CELL_STATE_SIZES = [1, 5, 10, 20, 30, 50, 100]
TIMESTEPS = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200]
NUM_LAYERS = [1, 2, 3, 4, 5]
NUM_EPOCHS = [1, 5, 10, 20, 50, 70, 100, 150, 200]

"""
Run each tuning separately and view the corresponding .txt file for RMSE evaluations.
Comment out the hyperparameter tuning functions that are not being used.
"""
def main(config=DEFAULT_CONFIG):
    hpt.tune_batch_size(BATCH_SIZES)
    hpt.tune_cell_state_size(CELL_STATE_SIZES, config.batch_size)
    hpt.tune_timesteps(TIMESTEPS, config.batch_size, config.cell_state_size)
    hpt.tune_num_layers(NUM_LAYERS, config.batch_size, config.cell_state_size, config.timestep)
    hpt.tune_num_epochs(NUM_EPOCHS, config.batch_size, config.cell_state_size, config.timestep, config.num_layers)

if __name__ == "__main__":
    main()
