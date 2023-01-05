import pickle

from src.models.cbow_dataset import CbowDataset


def load_data():
    with open('./data/training_data.pkl', 'rb') as file:
        data = pickle.load(file, encoding="latin1")

    vocabulary = data['chords_vocabulary']
    return (vocabulary, CbowDataset(data['train']), CbowDataset(data['valid']), CbowDataset(data['test']))
