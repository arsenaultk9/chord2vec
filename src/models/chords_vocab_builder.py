from typing import List

from src.models.chords_vocab import ChordsVocab
from src.models.chord import Chord
from src.models.notes import notes

def get_chord_and_notes(notes: List[str]):
    chord = Chord(notes)
    chord_notes = sorted(notes)
    return (chord, ''.join(chord_notes))

def get_possible_chords():
    possible_chords = dict()

    for first_note in notes:
        chord, chord_notes = get_chord_and_notes([first_note])
        possible_chords[chord_notes] = chord

        for second_note in notes:
            if first_note == second_note: # Chords must use different notes
                continue

            chord, chord_notes = get_chord_and_notes([first_note, second_note])
            possible_chords[chord_notes] = chord

            for third_note in notes:
                if first_note == third_note: # Chords must use different notes
                    continue

                if second_note == third_note: # Chords must use different notes
                    continue

                chord, chord_notes = get_chord_and_notes([first_note, second_note, third_note])
                possible_chords[chord_notes] = chord

    return possible_chords

def build_chords_vocab() -> ChordsVocab:
    suffixes_to_indexes = dict()
    indexes_ot_chords = dict()

    for index, (chord_notes, chord) in enumerate(get_possible_chords().items()):
        suffixes_to_indexes[chord_notes] = index
        indexes_ot_chords[index] = chord

    return ChordsVocab(suffixes_to_indexes, indexes_ot_chords)