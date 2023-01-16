from typing import List
from src.models.scale import Scale
from src.models.notes import notes


def get_scale_main_triads_filter(colors: List[str], chord_suffixes, scale: Scale, filter_color='red'):
    scale_chords = scale.get_main_triads()

    for chord_index, chord_suffix in enumerate(chord_suffixes):
        if chord_suffix in scale_chords:
            colors[chord_index] = filter_color

    return colors


def get_scale_all_triads_filter(colors: List[str], chord_suffixes, scale: Scale, filter_color='red'):
    scale_chords = scale.get_all_triads()

    for chord_index, chord_suffix in enumerate(chord_suffixes):
        if chord_suffix in scale_chords:
            colors[chord_index] = filter_color

    return colors



def get_scale_chord_degress(colors: List[str], chord_suffixes, chord_degree: int, mode: str, filter_color='red'):
    scales = map(lambda n : Scale(n, mode), notes)
    chord_degrees = list(map(lambda s : s.get_chord_degress_triad(chord_degree), scales))

    for chord_index, chord_suffix in enumerate(chord_suffixes):
        if chord_suffix in chord_degrees:
            colors[chord_index] = filter_color

    return colors
    