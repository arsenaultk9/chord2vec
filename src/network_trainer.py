import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import src.constants as constants
from src.networks.cbow_network import CbowNetwork
from src.results_aggregator import ResultsAggregator

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class NetworkTrainer:
    def __init__(self,
                 network: CbowNetwork,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 valid_data_loader: DataLoader):

        self.network = network

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.valid_data_loader = valid_data_loader

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

        results_aggregator = ResultsAggregator()

        for batch_idx, (x, y) in enumerate(self.train_data_loader):
            if len(x) < constants.BATCH_SIZE:
                continue  # Do not support smaller tensors that are not of batch size as first dimension

            self.optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            output = self.network(x)

            loss = self.get_loss(output, y)
            results_aggregator.aggregate_loss(loss.item())

            loss.backward()
            self.optimizer.step()

            accuracy = self.get_accuracy(output, y)
            results_aggregator.aggregate_accuracy(accuracy.item())

            if batch_idx % constants.BATCH_LOG_INTERVAL == 0 and batch_idx != 0:
                current_item = batch_idx * len(x)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tAverage Loss: {:.6f}\tAverage Right Predictions: {:.4f}'.format(
                    f"{epoch:03d}",
                    f"{current_item:04d}",
                    len(self.train_data_loader.dataset),
                    100. * batch_idx / len(self.train_data_loader),
                    results_aggregator.get_average_loss(batch_idx), 
                    results_aggregator.get_average_accuracy(batch_idx)
                    ))

        batch_idx = int(len(self.train_data_loader.dataset) / constants.BATCH_SIZE)
        results_aggregator.update_plots('train', batch_idx, epoch)
        

    def epoch_valid(self, epoch):
        self.network.eval()

        results_aggregator = ResultsAggregator()

        for batch_idx, (x, y) in enumerate(self.valid_data_loader):
            if len(x) < constants.BATCH_SIZE:
                continue  # Do not support smaller tensors that are not of batch size as first dimension

            self.optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            output = self.network(x)

            loss = self.get_loss(output, y)
            results_aggregator.aggregate_loss(loss.item())

            accuracy = self.get_accuracy(output, y)
            results_aggregator.aggregate_accuracy(accuracy.item())

            if batch_idx % constants.VALID_PREDICTION_SAMPLE_RATE == 0 and batch_idx != 0:
                current_item = batch_idx * len(x)

                print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss: {:.6f}\tAverage Right Predictions: {:.4f}'.format(
                    f"{epoch:03d}",
                    f"{current_item:04d}",
                    len(self.valid_data_loader.dataset),
                    100. * batch_idx / len(self.valid_data_loader),
                    results_aggregator.get_average_loss(batch_idx), 
                    results_aggregator.get_average_accuracy(batch_idx)
                    ))

        batch_idx = int(len(self.valid_data_loader.dataset) / constants.BATCH_SIZE)
        results_aggregator.update_plots('valid', batch_idx, epoch)


    def test(self):
        self.network.eval()

        results_aggregator = ResultsAggregator()

        for _, (x, y) in enumerate(self.test_data_loader):
            if len(x) < constants.BATCH_SIZE:
                continue  # Do not support smaller tensors that are not of batch size as first dimension

            self.optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            output = self.network(x)

            loss = self.get_loss(output, y)
            results_aggregator.aggregate_loss(loss.item())

            loss.backward()
            self.optimizer.step()

            accuracy = self.get_accuracy(output, y)
            results_aggregator.aggregate_accuracy(accuracy.item())

        current_item = int(len(self.test_data_loader.dataset) / constants.BATCH_SIZE)

        print('Test [{}/{}]\tAverage Loss: {:.6f}\tAverage Right Predictions: {:.4f}'.format(
            len(self.test_data_loader.dataset),
            len(self.test_data_loader.dataset),
            results_aggregator.get_average_loss(current_item), 
            results_aggregator.get_average_accuracy(current_item)
            ))