LOG_TENSORBOARD = True

INPUT_LENGTH = 9
MIDDLE_INPUT = 4

EMBED_DIMENSION = 25
EMBED_MAX_NORM = None

BATCH_LOG_INTERVAL = 50 #500 * 3 * 10
VALID_PREDICTION_SAMPLE_RATE = 20 #200  * 3 * 10
OPTIMIZER_ADAM_LR = 1.0
EPOCHS = 5
BATCH_SIZE = 1

SHUFFLE_DATA = False
APPLY_LR_SCHEDULER = True

SEQUENCE_GENERATION_LENGTH = 18

TRAINING_DATA_PATH = './data/training_data.pkl'

# Data augmentation
APPLY_SCALE_AUGMENTATION = True
