import pickle

import src.constants as constants

from src.models.generation_dataset import GenerationDataset

def load_data():
    with open(constants.EMBEDDING_TRAINING_DATA_PATH, 'rb') as file:
        return pickle.load(file, encoding="latin1")


def load_generation_data():
    data = load_data()

    vocabulary = data['chords_vocabulary']
    return (vocabulary, GenerationDataset(data['train']), GenerationDataset(data['valid']), GenerationDataset(data['test']))
