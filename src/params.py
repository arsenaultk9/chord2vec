class Params:
    def __init__(self,
                 WINDOW_SLIDE_RANGE,
                 EMBEDDING_TRAINING_DATA_PATH,
                 GENERATION_TRAINING_DATA_PATH,
                 EMBEDDING_MODEL_PATH,
                 SHUFFLE_DATA_RANDOM_FOREST,
                 EMBED_DATA_RANDOM_FOREST):
        self.WINDOW_SLIDE_RANGE = WINDOW_SLIDE_RANGE
        self.EMBEDDING_TRAINING_DATA_PATH = EMBEDDING_TRAINING_DATA_PATH
        self.GENERATION_TRAINING_DATA_PATH = GENERATION_TRAINING_DATA_PATH
        self.EMBEDDING_MODEL_PATH = EMBEDDING_MODEL_PATH
        self.SHUFFLE_DATA_RANDOM_FOREST = SHUFFLE_DATA_RANDOM_FOREST
        self.EMBED_DATA_RANDOM_FOREST = EMBED_DATA_RANDOM_FOREST


global params
params = Params(1, './data/training_data.pkl', './data/training_data_small.pkl', 'result_model/cbow_network.pt', True, True)


def set_params(new_params: Params):
    global params
    params = new_params


def get_params():
    global params
    return params
