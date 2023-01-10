from typing import Dict

from src.models.chord import Chord


class ChordsVocab:
    def __init__(self, suffixes_to_indexes: Dict[str, int], indexes_to_chords: Dict[int, Chord]):
        self.suffixes_to_indexes = suffixes_to_indexes
        self.indexes_to_chords = indexes_to_chords