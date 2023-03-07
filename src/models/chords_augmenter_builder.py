from typing import Dict, List
from src.models.chords_augmenter import ChordsAugmenter
from src.models.chords_vocab import ChordsVocab
from src.models.notes import notes
from src.models.chord import Chord

def get_notes_ordered_by_interval_skip(note: str):
    note_pos = notes.index(note)

    # Skip current note and extend notes so that current note would be first note of array
    return notes[note_pos+1:] + notes[:note_pos]

def get_possible_note_augmentations():
    possible_note_augmentations = dict()

    for note in notes:
        possible_note_augmentations[note] = []

        notes_ordered_from_note = get_notes_ordered_by_interval_skip(note)

        for note_augmentation in notes_ordered_from_note:
            possible_note_augmentations[note].append(note_augmentation)

    return possible_note_augmentations

def get_chord_augmentation_indexes(chords_vocab: ChordsVocab, chord: Chord, possible_note_augmentations: Dict[str, List[str]]):
    chord_note_augmentations = []
    chord_note_combos = []

    for note in chord.note_suffixes:
        note_augmentations = possible_note_augmentations[note] 
        chord_note_combos.append(note_augmentations)

    for note_combo in zip(*chord_note_combos):
        chord_suffix = ''.join(sorted(note_combo))
        chord_index = chords_vocab.suffixes_to_indexes[chord_suffix]

        chord_note_augmentations.append(chord_index)

    return chord_note_augmentations


def build_chords_augmenter(chords_vocab: ChordsVocab) -> ChordsAugmenter:
    chord_augmentations = dict()
    possible_note_augmentations = get_possible_note_augmentations()

    for chord_index, chord in chords_vocab.indexes_to_chords.items():
        chord_augmentation_indexes = get_chord_augmentation_indexes(chords_vocab, chord, possible_note_augmentations)
        chord_augmentations[chord_index] = chord_augmentation_indexes

    return ChordsAugmenter(chord_augmentations)