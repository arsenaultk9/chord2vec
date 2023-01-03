import torch
from torch.utils.data import DataLoader

import src.constants as constants
from src.data_loader import load_data
from src.networks.cbow_network import CbowNetwork
from src.network_trainer import NetworkTrainer


use_cuda = False # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

vocabulary, cbow_dataset = load_data()

train_data_loader = DataLoader(cbow_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)

network = CbowNetwork(len(vocabulary.values())).to(device)
trainer = NetworkTrainer(network, train_data_loader)

for epoch in range(1, constants.EPOCHS + 1):
    trainer.epoch_train(epoch)
