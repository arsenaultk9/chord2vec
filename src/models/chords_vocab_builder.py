from typing import Dict, List
from src.models.notes import notes

def get_chord_notes(notes: List[str]):
    chord = sorted(notes)
    return ''.join(chord)

def get_possible_chords():
    possible_chords = set()

    for first_note in notes:
        chord_notes = get_chord_notes([first_note])
        possible_chords.add(chord_notes)

        for second_note in notes:
            if first_note == second_note: # Chords must use different notes
                continue

            chord_notes = get_chord_notes([first_note, second_note])
            possible_chords.add(chord_notes)

            for third_note in notes:
                if first_note == third_note: # Chords must use different notes
                    continue

                if second_note == third_note: # Chords must use different notes
                    continue

                chord_notes = get_chord_notes([first_note, second_note, third_note])
                possible_chords.add(chord_notes)

    return possible_chords

def build_chords_vocab() -> Dict[int, str]:
    vocab = dict()

    for index, chord in enumerate(get_possible_chords()):
        vocab[chord] = index

    return vocab