import torch

import src.constants as constants
from src.networks.cbow_network import CbowNetwork

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class NetworkCbowGenerator:
    def __init__(self, network: CbowNetwork):
        self.network = network

    def generate_sequence(self, x_sequence: torch.Tensor):
        generated_sequence = []

        x_sequence = x_sequence.to(device)

        for x_item in x_sequence[0]:
            generated_sequence.append(x_item.item())

        for _ in range(constants.SEQUENCE_GENERATION_LENGTH):
            y_pred = self.network(x_sequence)
            y_class_pred = y_pred.argmax(dim=1)

            generated_sequence.append(y_class_pred[0].item())

            new_x_sequence = x_sequence[:, 1:constants.CBOW_INPUT_LENGTH]
            add_y_inputs = torch.reshape(y_class_pred, (constants.BATCH_SIZE, 1))
            x_sequence = torch.concat((new_x_sequence, add_y_inputs), 1)

        return generated_sequence
