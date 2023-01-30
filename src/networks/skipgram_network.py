import torch
import torch.nn as nn
import src.constants as constants

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class SkipgramNetwork(nn.Module):
    """
    Implementation of Skipgram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(SkipgramNetwork, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=constants.EMBED_DIMENSION,
            max_norm=constants.EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=constants.EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def get_initial_hidden_context(self):
        h = torch.zeros((1, constants.BATCH_SIZE, constants.LSTM_HIDDEN_SIZE)).to(
            device)  # 1 is for num_layers * 1 for unidirectional lstm
        c = torch.zeros(
            (1, constants.BATCH_SIZE, constants.LSTM_HIDDEN_SIZE)).to(device)

        return (h, c)

    def forward(self, inputs, _):
        x = self.embeddings(inputs)
        x = self.linear(x)
        return x.transpose(1, 2), _