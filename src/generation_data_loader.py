import pickle

import src.constants as constants

from src.params import get_params
from src.models.generation_dataset import GenerationDataset
from src.generation_dataset_builder import get_input_and_targets


def load_data():
    with open(get_params().generation_training_data_path, 'rb') as file:
        return pickle.load(file, encoding="latin1")


def load_generation_data():
    data = load_data()

    vocabulary = data['chords_vocabulary']
    return (vocabulary, GenerationDataset(data['train']), GenerationDataset(data['valid']), GenerationDataset(data['test']))


def load_random_forest_data():
    data = load_data()

    vocabulary = data['chords_vocabulary']
    _, X_train, y_train = get_input_and_targets(data['train'] + data['valid'])
    _, X_test, y_test = get_input_and_targets(data['test'])

    return (vocabulary, X_train, y_train, X_test, y_test)

