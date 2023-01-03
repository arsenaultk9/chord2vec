import pickle

from src.models.cbow_dataset import CbowDataset

def load_data():
    with open('./data/training_data.pkl', 'rb') as file:
        data = pickle.load(file, encoding="latin1")

    return CbowDataset(data['data'], data['chords_vocabulary'])