from typing import Dict, List

import torch

import src.constants as constants

def get_input_and_targets(sequences: List[List[int]]):
    song_indexes = []
    Xs = []
    Ys = []

    for song_index, song in enumerate(sequences):
        for start_pos in range(0, len(song) - constants.INPUT_LENGTH - 1, constants.INPUT_LENGTH):
            song_indexes.append(song_index)

            input_middle_pos = start_pos + constants.MIDDLE_INPUT
            input_end_pos = start_pos + constants.INPUT_LENGTH

            x_target_form = [song[input_middle_pos]] * (constants.INPUT_LENGTH - 1) # This is good because for python [x:y] and [y] y is upper bound excluded in first form.
            Xs.append(x_target_form)

            y =  song[start_pos:input_middle_pos] + song[input_middle_pos+1:input_end_pos] 
            Ys.append(y)

    return (song_indexes, Xs, Ys)

def get_training_data(sequences: List[List[int]]):
    song_indexes, Xs, Ys = get_input_and_targets(sequences)

    X = torch.tensor(Xs, dtype=torch.long)
    Y = torch.tensor(Ys, dtype=torch.long)

    return (song_indexes, X, Y)
    