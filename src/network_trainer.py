import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import src.constants as constants
from src.networks.cbow_network import CbowNetwork

use_cuda = torch.cuda.is_available()
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

    def get_accuracy(self, model_output, y_target):
        output_class_pred = model_output.argmax(dim=1)
        equal_samples = torch.eq(output_class_pred, y_target)

        return equal_samples.sum() / constants.BATCH_SIZE


    def epoch_train(self, epoch):
        self.network.train()

        losses = []
        accuracies = []

        for batch_idx, (x, y) in enumerate(self.train_data_loader):
            if len(x) < constants.BATCH_SIZE:
                continue  # Do not support smaller tensors that are not of batch size as first dimension

            self.optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            output = self.network(x)

            loss = self.get_loss(output, y)
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()

            accuracy = self.get_accuracy(output, y)
            accuracies.append(accuracy.item())

            if batch_idx % constants.BATCH_LOG_INTERVAL == 0 and batch_idx != 0:
                current_item = batch_idx * len(x)
                average_loss = sum(losses) / batch_idx
                average_accuracy = sum(accuracies) / batch_idx

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tAverage Loss: {:.6f}\tAverage Right Predictions: {:.4f}'.format(
                    f"{epoch:03d}",
                    f"{current_item:04d}",
                    len(self.train_data_loader.dataset),
                    100. * batch_idx / len(self.train_data_loader),
                    average_loss, 
                    average_accuracy
                    ))