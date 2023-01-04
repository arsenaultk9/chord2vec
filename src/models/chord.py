class Chord:
    def __init__(self, note_suffixes):
        self.note_suffixes = note_suffixes


    def __str__(self):
        notes_sorted = sorted(self.note_suffixes)
        note_suffixes_concat = f'[ {", ".join(notes_sorted)} ]'
        return note_suffixes_concat

    def __repr__(self):
        return self.__str__()


    @classmethod
    def create_from_json(cls, chord_data_json):
        note_suffixes = []

        for harmony_note in chord_data_json['HarmonyNotes']:
            note_suffixes.append(harmony_note['Suffix'])

        return cls(note_suffixes)