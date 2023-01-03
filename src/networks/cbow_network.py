import torch.nn as nn
import src.constants as constants

class CbowNetwork(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(CbowNetwork, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=constants.EMBED_DIMENSION,
            max_norm=constants.EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=constants.EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x