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
        self.embeddings.weight.requires_grad = False

        self.lstm = nn.LSTM(constants.EMBED_DIMENSION, 128,
                            bidirectional=False, batch_first=True)

        self.linear = nn.Linear(
            in_features=128,
            out_features=vocab_size,
        )

    def forward(self, inputs):
        x = self.embeddings(inputs)
        (x, h)= self.lstm(x)
        x = self.linear(x)

        return x[:,-1]
