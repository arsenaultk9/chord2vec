from typing import Dict
from src.models.notes import notes

def get_possible_chords():
    possible_chords = set()

    for first_note in notes:
        for second_note in notes:
            if first_note == second_note: # Chords must use different notes
                continue

            for third_note in notes:
                if first_note == third_note: # Chords must use different notes
                    continue

                if second_note == third_note: # Chords must use different notes
                    continue

                chord = sorted([first_note, second_note, third_note])
                chord_notes = ''.join(chord)

                possible_chords.add(chord_notes)

    return possible_chords

def build_chords_vocab() -> Dict[int, str]:
    vocab = dict()

    for index, chord in enumerate(get_possible_chords()):
        vocab[index] = chord

    return vocab