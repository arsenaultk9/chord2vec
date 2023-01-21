import pickle

import src.constants as constants

from src.models.cbow_dataset import CbowDataset
from src.models.skipgram_dataset import SkipgramDataset

def load_data():
    with open(constants.TRAINING_DATA_PATH, 'rb') as file:
        return pickle.load(file, encoding="latin1")


def load_cbow_data():
    data = load_data()

    vocabulary = data['chords_vocabulary']
    return (vocabulary, CbowDataset(data['train']), CbowDataset(data['valid']), CbowDataset(data['test']))


def load_skipgram_data():
    data = load_data()

    vocabulary = data['chords_vocabulary']
    return (vocabulary, SkipgramDataset(data['train']), SkipgramDataset(data['valid']), SkipgramDataset(data['test']))
