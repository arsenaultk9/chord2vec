from typing import List

from src.models.note_info import NoteInfo
from src.models.chords_vocab import ChordsVocab
from src.models.notes import notes_to_pitch


def generate_note_info(generated_sequence: List[int], vocabulary: ChordsVocab):
    note_infos = []

    cur_pos = 0

    for generated_item in generated_sequence:
        chord = vocabulary.indexes_to_chords[generated_item]

        for chord_note in chord.note_suffixes:
            note_pitch = notes_to_pitch[chord_note]
            note_info = NoteInfo.create(cur_pos, cur_pos + 2, note_pitch)

            note_infos.append(note_info)

        cur_pos += 2

    return [note_infos]