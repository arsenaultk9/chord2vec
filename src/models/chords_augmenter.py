from typing import Dict, List, Tuple

from src.models.chord import Chord


class ChordsAugmenter:
    def __init__(self, chord_augmentations: Dict[int, List[int]]):
        self.chord_augmentations = chord_augmentations