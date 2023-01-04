from typing import Dict, List

import numpy as np
import torch

import src.constants as constants

def get_input_and_targets(sequences: List[List[int]]):
    Xs = []
    Ys = []

    for song in sequences:
        for start_pos in range(len(song) - constants.CBOW_INPUT_LENGTH - 1):
            input_end_pos = start_pos + constants.CBOW_INPUT_LENGTH
            target_pos = input_end_pos + 1

            x_target_form = song[start_pos:input_end_pos]
            Xs.append(x_target_form)

            y = song[target_pos]
            Ys.append(y)

    return (Xs, Ys)

def get_training_data(sequences: List[List[int]]):
    Xs, Ys = get_input_and_targets(sequences)

    X = torch.tensor(Xs, dtype=torch.long)
    Y = torch.tensor(Ys, dtype=torch.long)

    return (X, Y)
    