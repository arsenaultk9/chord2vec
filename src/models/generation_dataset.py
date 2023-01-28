from typing import Dict, List
from torch.utils.data import Dataset

from src.generation_dataset_builder import get_training_data


class GenerationDataset(Dataset):
    def __init__(self, sequences: List[List[int]]):
        self.song_indexes, self.X, self.Y = get_training_data(sequences)

        self.length = self.X.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        song_index = self.song_indexes[idx]
        X = self.X[idx]
        Y = self.Y[idx]

        return (song_index, X, Y)
