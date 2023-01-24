import torch.nn as nn
import src.constants as constants

class LstmEmbeddingNetwork(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int, embedding_weights):
        super(LstmEmbeddingNetwork, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embedding_weights)
        self.linear = nn.Linear(
            in_features=constants.EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x