from typing import Dict, List

import numpy as np
import torch

import src.constants as constants

def get_one_hot_encode_form(target_form: List[int], vocabulary: Dict[int, str]):
    vocab_length = len(vocabulary.values())
    x_input_from = np.zeros((constants.CBOW_INPUT_LENGTH, vocab_length))

    for pos, cur_x in enumerate(target_form):
        x_input_from[pos, cur_x] = 1

    return x_input_from

def get_input_and_targets(sequences: List[List[int]], vocabulary: Dict[int, str]):
    Xs = []
    Ys = []

    for song in sequences:
        for start_pos in range(len(song) - constants.CBOW_INPUT_LENGTH - 1):
            input_end_pos = start_pos + constants.CBOW_INPUT_LENGTH
            target_pos = input_end_pos + 1

            x_target_form = song[start_pos:input_end_pos]
            x_input = get_one_hot_encode_form(x_target_form, vocabulary)
            Xs.append(x_input)

            y = song[target_pos]
            Ys.append(y)

    return (Xs, Ys)

def get_training_data(sequences: List[List[int]], vocabulary: Dict[int, str]):
    Xs, Ys = get_input_and_targets(sequences, vocabulary)

    X = torch.tensor(Xs, dtype=torch.float32)
    Y = torch.tensor(Ys, dtype=torch.float32)

    return (X, Y)
    