import torch
from torch.utils.data import DataLoader

import src.constants as constants

from src.params import get_params, set_params, Params
from src.embedding_data_loader import load_cbow_data, load_skipgram_data
from src.networks.cbow_network import CbowNetwork
from src.networks.skipgram_network import SkipgramNetwork
from src.network_trainer import NetworkTrainer


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def train_embedding(params: Params):
    set_params(params)

    vocabulary, cbow_train_dataset, cbow_valid_dataset, cbow_test_dataset = load_cbow_data()

    train_data_loader = DataLoader(cbow_train_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
    valid_data_loader = DataLoader(cbow_valid_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
    test_data_loader = DataLoader(cbow_test_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)

    network = CbowNetwork(len(vocabulary.suffixes_to_indexes.values())).to(device)
    trainer = NetworkTrainer(network, train_data_loader, valid_data_loader, test_data_loader, constants.EMBEDDING_EPOCHS, is_dynamic_lr_scheduler=False)

    for epoch in range(1, constants.EMBEDDING_EPOCHS + 1):
        trainer.epoch_train(epoch)
        trainer.epoch_valid(epoch)

    test_accuracy = trainer.test()

    # Turn off training mode & switch to model evaluation
    network.eval()

    # === Save model for production use ===
    (_, x_sequence, y_pred) = cbow_train_dataset[0:constants.BATCH_SIZE]
    (h, c) = network.get_initial_hidden_context()

    traced_script_module = torch.jit.trace(network.forward, (x_sequence.to(device), (h, c)))
    traced_script_module.save(get_params().embedding_model_path)

    return test_accuracy

if __name__ == "__main__":
    train_embedding(get_params())