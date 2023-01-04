from typing import Dict

from src.models.chord import Chord


class ChordsVocab:
    def __init__(self, suffixes_to_indexes: Dict[str, int], indexes_ot_chords: Dict[int, Chord]):
        self.suffixes_to_indexes = suffixes_to_indexes
        self.indexes_ot_chords = indexes_ot_chords