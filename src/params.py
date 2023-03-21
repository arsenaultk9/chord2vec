class Params:
    def __init__(self,
                 window_slide_range = 1,
                 embedding_training_data_path = './data/training_data.pkl',
                 generation_training_data_path = './data/training_data_small.pkl',
                 embedding_model_path = 'result_model/cbow_network.pt',
                 shuffle_data_random_forest = True,
                 embed_data= True):
        self.window_slide_range = window_slide_range
        self.embedding_training_data_path = embedding_training_data_path
        self.generation_training_data_path = generation_training_data_path
        self.embedding_model_path = embedding_model_path
        self.shuffle_data_random_forest = shuffle_data_random_forest
        self.embed_data = embed_data


global params
params = Params()


def set_params(new_params: Params):
    global params
    params = new_params


def get_params():
    global params
    return params
