LOG_TENSORBOARD = False

INPUT_LENGTH = 5
MIDDLE_INPUT = 2
WINDOW_SLIDE_RANGE = 1

EMBED_DIMENSION = 25
EMBED_MAX_NORM = 1

LSTM_HIDDEN_SIZE = 128

BATCH_LOG_INTERVAL = 50 # 500 * 10 #500 * 3 * 10
VALID_PREDICTION_SAMPLE_RATE = 20 # 200 * 10 #200  * 3 * 10
OPTIMIZER_ADAM_LR = 1.

EMBEDDING_EPOCHS = 5
GENERATION_EPOCHS = 16

BATCH_SIZE = 1

APPLY_LR_SCHEDULER = True

SEQUENCE_GENERATION_LENGTH = 18

EMBEDDING_TRAINING_DATA_PATH = './data/training_data.pkl'
GENERATION_TRAINING_DATA_PATH = './data/training_data_small.pkl'

SHUFFLE_DATA_RANDOM_FOREST = True
EMBED_DATA_RANDOM_FOREST = True

# Data augmentation
APPLY_DATA_AUGMENTATION = True
DATA_AUGMENTATION_COUNT = 4
