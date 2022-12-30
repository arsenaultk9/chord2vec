class Chord:
    def __init__(self, chord_data_json):
        self.note_suffixes = []

        for harmony_note in chord_data_json['HarmonyNotes']:
            self.note_suffixes.append(harmony_note['Suffix'])


    def __str__(self):
        notes_sorted = sorted(self.note_suffixes)
        note_suffixes_concat = f'[ {", ".join(notes_sorted)} ]'
        return note_suffixes_concat

    def __repr__(self):
        return self.__str__()