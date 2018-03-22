# These are the hyperparameters chosen after performing hyperparameter tuning.
class Config():
    batch_size=512
    cell_state_size=30
    timestep=50
    num_layers=3
    num_epochs=50

DEFAULT_CONFIG = Config()
