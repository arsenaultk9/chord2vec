import torch.nn as nn
import src.constants as constants


class LstmVanillaNetwork(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """

    def __init__(self, vocab_size: int, embedding_weights): # Keep param for ease of switch of generation networks.
        super(LstmVanillaNetwork, self).__init__()

        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=constants.EMBED_DIMENSION,
            max_norm=constants.EMBED_MAX_NORM,
        )
        self.lstm = nn.LSTM(constants.EMBED_DIMENSION, 128,
                            bidirectional=False, batch_first=True)

        self.linear = nn.Linear(
            in_features=128,
            out_features=vocab_size,
        )

    def forward(self, inputs):
        x = self.embeddings(inputs)
        (x, h)= self.lstm(x)
        x = x[:,-1] # Only keep last output cell
        x = self.linear(x)

        return x
