import pickle

import src.constants as constants

from src.params import get_params
from src.models.cbow_dataset import CbowDataset
from src.models.skipgram_dataset import SkipgramDataset

def load_data():
    with open(get_params().embedding_training_data_path, 'rb') as file:
        return pickle.load(file, encoding="latin1")


def load_cbow_data():
    data = load_data()

    vocabulary = data['chords_vocabulary']
    return (vocabulary, CbowDataset(data['train']), CbowDataset(data['valid']), CbowDataset(data['test']))


def load_skipgram_data():
    data = load_data()

    vocabulary = data['chords_vocabulary']
    return (vocabulary, SkipgramDataset(data['train']), SkipgramDataset(data['valid']), SkipgramDataset(data['test']))
