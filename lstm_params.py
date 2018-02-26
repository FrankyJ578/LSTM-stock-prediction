DATA_DIR = "data"
LOG_DIR = "logs"
MODEL_DIR = "models"

class LSTMParams():
    window_size = 1
    num_steps = 30
    layer_size = 128
    num_layers = 1
    keep_prob = 0.9
    init_learning_rate = 0.001
    learning_rate_decay = 0.99
    batch_size = 64
    init_epoch = 5
    max_epoch = 50

PARAMS = LSTMParams()
