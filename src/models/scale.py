from typing import List
from src.models.notes import notes

# Interval position by mode
mode_intervals = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10]
}

def get_scale_notes(root: str, mode=str):
    scale_notes = []

    mode_interval = mode_intervals[mode]
    start_index = notes.index(root)

    for interval_pos in mode_interval:
        cur_index = start_index + interval_pos
        cur_index = cur_index if cur_index < 12 else cur_index - 12
        
        note = notes[cur_index]
        scale_notes.append(note)


    return scale_notes

def get_note_in_bound(notes: List[str], note_index: int):
    if note_index < len(notes):
        return notes[note_index]

    index_from_bound = note_index - len(notes)
    return notes[index_from_bound]

class Scale:
    def __init__(self, root: str, mode='major'):
        self.root = root
        self.mode = mode
        self.notes = get_scale_notes(root, mode)


    def get_main_triads(self):
        triad_chords = []

        for note_index in range(len(self.notes)):
            note_a = get_note_in_bound(self.notes, note_index)
            note_b = get_note_in_bound(self.notes, note_index + 2)
            note_c = get_note_in_bound(self.notes, note_index + 4)

            chord_suffix = ''.join(sorted([note_a, note_b, note_c]))
            triad_chords.append(chord_suffix)

        return triad_chords


    def get_all_triads(self):
        triad_chords = []

        for note_a in self.notes:
            chord_suffix = ''.join(sorted([note_a]))
            triad_chords.append(chord_suffix)

            for note_b in self.notes:
                if note_a == note_b:
                    continue

                chord_suffix = ''.join(sorted([note_a, note_b]))
                triad_chords.append(chord_suffix)

                for note_c in self.notes:
                    if note_a == note_c or note_b == note_c:
                        continue

                    chord_suffix = ''.join(sorted([note_a, note_b, note_c]))
                    triad_chords.append(chord_suffix)

        return triad_chords


    def get_chord_degress_triad(self, chord_degree: int):
        note_a = get_note_in_bound(self.notes, chord_degree)
        note_b = get_note_in_bound(self.notes, chord_degree + 2)
        note_c = get_note_in_bound(self.notes, chord_degree + 4)

        return ''.join(sorted([note_a, note_b, note_c]))


    def __str__(self):
        return f'root: {self.root}, mode: {self.mode}, notes: [ {", ".join(self.notes)} ]'

    def __repr__(self):
        return self.__str__()