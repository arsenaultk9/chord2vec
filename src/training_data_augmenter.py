import random
from typing import List, Dict

import src.constants as constants

from src.models.chords_vocab import ChordsVocab
from src.models.chords_augmenter_builder import build_chords_augmenter
from src.models.chords_augmenter import ChordsAugmenter
from src.models.notes import notes


def get_chord_counts(training_data: List[int], chords_vocab: ChordsVocab) -> Dict[int, int]:
    chord_counts = dict()

    for chord_index in chords_vocab.indexes_to_chords.keys():
        chord_counts[chord_index] = 0

    for song in training_data:
        for chord_index in song:
            chord_counts[chord_index] += 1

    return chord_counts

def get_possible_augmentations(chords_augmenter: ChordsAugmenter, song: List[int]) -> Dict[int, List[int]]:
    song_augmentations = dict()

    for augmentation_index in range(len(notes) - 1):
        song_augmentations[augmentation_index] = []

    for song_chord in song:
        chord_augmentations = chords_augmenter.chord_augmentations[song_chord]

        for augmentation_index, chord_augmentation in enumerate(chord_augmentations):
            song_augmentations[augmentation_index].append(chord_augmentation)

    return song_augmentations

def get_best_augmentations(song_augmentations: Dict[int, List[int]], chord_counts: Dict[int, int]) -> List[List[int]]:
    best_augmentations = []
    song_augmentation_scores = dict()

    for augmentation_index, song_augmentation in song_augmentations.items():
        score = sum(map(lambda ci : chord_counts[ci], song_augmentation))
        song_augmentation_scores[augmentation_index] = score

    # Get augmentations that have the least occurence in chord_counts
    for augmentation_index, _ in sorted(song_augmentation_scores.items(), key=lambda item: item[1]):
        if(len(best_augmentations) >= constants.DATA_AUGMENTATION_COUNT):
            break

        best_augmentation = song_augmentations[augmentation_index]
        best_augmentations.append(best_augmentation)

        # Upgrade chord count score
        for chord_index in best_augmentation:
            chord_counts[chord_index] += 1


    return best_augmentations

def augment_training_data(training_data: List[List[int]], chords_vocab: ChordsVocab) -> List[List[int]]:
    if not constants.APPLY_DATA_AUGMENTATION:
        return training_data
    
    chords_augmenter = build_chords_augmenter(chords_vocab)

    augmented_training_data = training_data.copy()
    chord_counts = get_chord_counts(training_data, chords_vocab)

    for song in training_data:
        possible_augmentations = get_possible_augmentations(chords_augmenter, song)
        best_augmentations = get_best_augmentations(possible_augmentations, chord_counts)
        augmented_training_data += best_augmentations

    random.shuffle(augmented_training_data)
    return augmented_training_data