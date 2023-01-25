LOG_TENSORBOARD = True

INPUT_LENGTH = 5
MIDDLE_INPUT = 2

EMBED_DIMENSION = 25
EMBED_MAX_NORM = None

BATCH_LOG_INTERVAL = 500 #500 * 3 * 10
VALID_PREDICTION_SAMPLE_RATE = 200 #200  * 3 * 10
OPTIMIZER_ADAM_LR = 1.

EMBEDDING_EPOCHS = 5
GENERATION_EPOCHS = 16

BATCH_SIZE = 1

SHUFFLE_DATA = False
APPLY_LR_SCHEDULER = True

SEQUENCE_GENERATION_LENGTH = 18

EMBEDDING_TRAINING_DATA_PATH = './data/training_data.pkl'
GENERATION_TRAINING_DATA_PATH = './data/training_data_small.pkl'

# Data augmentation
APPLY_SCALE_AUGMENTATION = False
