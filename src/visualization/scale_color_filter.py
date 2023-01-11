from src.models.scale import Scale


def get_scale_triads_filter(chord_suffixes, scale: Scale):
    colors = []
    scale_chords = scale.get_main_triads()

    for chord_suffix in chord_suffixes:
        color = 'red' if chord_suffix in scale_chords else 'lightgrey'
        colors.append(color)

    return colors