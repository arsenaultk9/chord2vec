from src.models.scale import Scale
from src.models.notes import notes


def get_scale_main_triads_filter(chord_suffixes, scale: Scale):
    colors = []
    scale_chords = scale.get_main_triads()

    for chord_suffix in chord_suffixes:
        color = 'red' if chord_suffix in scale_chords else 'lightgrey'
        colors.append(color)

    return colors


def get_scale_all_triads_filter(chord_suffixes, scale: Scale):
    colors = []
    scale_chords = scale.get_all_triads()

    for chord_suffix in chord_suffixes:
        color = 'red' if chord_suffix in scale_chords else 'lightgrey'
        colors.append(color)

    return colors



def get_scale_chord_degress(chord_suffixes, chord_degree: int, mode: str):
    colors = []

    scales = map(lambda n : Scale(n, mode), notes)
    chord_degrees = list(map(lambda s : s.get_chord_degress_triad(chord_degree), scales))

    print(chord_degrees)

    for chord_suffix in chord_suffixes:
        color = 'red' if chord_suffix in chord_degrees else 'lightgrey'
        colors.append(color)

    return colors
    