import torch
import torch.nn as nn
import src.constants as constants

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class LstmEmbeddingNetwork(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """

    def __init__(self, vocab_size: int, embedding_weights):
        super(LstmEmbeddingNetwork, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(embedding_weights)
        self.embeddings.weight.requires_grad = False

        self.lstm = nn.LSTM(constants.EMBED_DIMENSION, constants.LSTM_HIDDEN_SIZE,
                            bidirectional=False, batch_first=True)

        self.linear = nn.Linear(
            in_features=constants.LSTM_HIDDEN_SIZE,
            out_features=vocab_size,
        )

    def get_initial_hidden_context(self):
        h = torch.zeros((1, constants.BATCH_SIZE, constants.LSTM_HIDDEN_SIZE)).to(
            device)  # 1 is for num_layers * 1 for unidirectional lstm
        c = torch.zeros(
            (1, constants.BATCH_SIZE, constants.LSTM_HIDDEN_SIZE)).to(device)

        return (h, c)

    def forward(self, inputs, h_c_tupple):
        x = self.embeddings(inputs)
        (x, (h, c)) = self.lstm(x, h_c_tupple)
        x = x[:, -1]  # Only keep last output cell
        x = self.linear(x)

        return x, (h, c)
