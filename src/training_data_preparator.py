from typing import Dict, List

from src.models.chord import Chord

def get_song_training_data(song_chords: List[Chord], chords_vocabulary: Dict[str, int]):
    song_training_data = []

    for song_chord in song_chords:
        chord_notes = sorted(song_chord.note_suffixes)
        chord_notes_str = ''.join(chord_notes)

        chord_index = chords_vocabulary[chord_notes_str]
        song_training_data.append(chord_index)

    return song_training_data


def get_training_data(all_song_chords: List[List[Chord]], chords_vocabulary: Dict[str, int]) -> List[List[int]]:
    training_data = []

    for song_chords in all_song_chords:
        song_training_data = get_song_training_data(song_chords, chords_vocabulary)
        training_data.append(song_training_data)

    return training_data