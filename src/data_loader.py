import pickle

from src.models.cbow_dataset import CbowDataset

def load_data():
    with open('./data/training_data.pkl', 'rb') as file:
        return pickle.load(file, encoding="latin1")


def load_cbow_data():
    data = load_data()

    vocabulary = data['chords_vocabulary']
    return (vocabulary, CbowDataset(data['train']), CbowDataset(data['valid']), CbowDataset(data['test']))


def load_skip_gram_data():
    data = load_data()

    vocabulary = data['chords_vocabulary']
    return (vocabulary, CbowDataset(data['train']), CbowDataset(data['valid']), CbowDataset(data['test']))
