from typing import List

from src.models.chord import Chord


def extract_chords(song) -> List[Chord]:
    song_chords = []

    for chordMeasure in song['ChordsMeasures']:
        for chordData in chordMeasure['Chords']:
            chord = Chord(chordData)

            if len(song_chords) > 0 and str(chord) == str(song_chords[-1]):
                continue

            song_chords.append(chord)

    return song_chords