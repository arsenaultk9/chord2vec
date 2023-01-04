from typing import List

from src.models.chords_vocab import ChordsVocab
from src.models.chord import Chord

def get_song_training_data(song_chords: List[Chord], chords_vocabulary: ChordsVocab):
    song_training_data = []

    for song_chord in song_chords:
        chord_notes = sorted(song_chord.note_suffixes)
        chord_notes_str = ''.join(chord_notes)

        chord_index = chords_vocabulary.suffixes_to_indexes[chord_notes_str]
        song_training_data.append(chord_index)

    return song_training_data


def get_training_data(all_song_chords: List[List[Chord]], chords_vocabulary: ChordsVocab) -> List[List[int]]:
    training_data = []

    for song_chords in all_song_chords:
        song_training_data = get_song_training_data(song_chords, chords_vocabulary)
        training_data.append(song_training_data)

    return training_data