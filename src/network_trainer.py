import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import src.constants as constants
from src.networks.cbow_network import CbowNetwork

use_cuda = False # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class NetworkTrainer:
    def __init__(self,
                 network: CbowNetwork,
                 train_data_loader: DataLoader):

        self.network = network
        self.train_data_loader = train_data_loader

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

    def get_loss(self, model_output, y_target):
        return self.loss_function(model_output, y_target)

    def epoch_train(self, epoch):
        self.network.train()

        losses = []

        for batch_idx, (x, y) in enumerate(self.train_data_loader):
            if len(x) < constants.BATCH_SIZE:
                continue  # Do not support smaller tensors that are not of batch size as first dimension

            self.optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            output = self.network(x)

            loss = self.get_loss(output, y)

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if batch_idx % constants.BATCH_LOG_INTERVAL == 0 and batch_idx != 0:
                current_item = batch_idx * len(x)
                average_loss = sum(losses) / current_item

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tAverage Loss: {:.6f}\t'.format(
                    f"{epoch:03d}",
                    f"{current_item:04d}",
                    len(self.train_data_loader.dataset),
                    100. * batch_idx / len(self.train_data_loader),
                    average_loss, 
                    ))