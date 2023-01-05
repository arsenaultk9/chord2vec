from torch.utils.tensorboard import SummaryWriter
import src.constants as constants

if constants.LOG_TENSORBOARD:
    writer = SummaryWriter()

class ResultsAggregator:
    def __init__(self):
        self.total_losses = []
        self.total_accuracies = []

    def aggregate_loss(self, current_batch_loss):
        self.total_losses.append(current_batch_loss)

    def aggregate_accuracy(self, average_accuracy):
        self.total_accuracies.append(average_accuracy)

    def get_average_loss(self, item_count):
        total_loss = sum(self.total_losses)
        return total_loss / item_count

    def get_average_accuracy(self, item_count):
        total_accuracies = sum(self.total_accuracies)
        average_right_predictions = total_accuracies / item_count

        return average_right_predictions


    def update_plots(self, mode, item_count, epoch):
        if not constants.LOG_TENSORBOARD:
            return

        average_loss = self.get_average_loss(item_count)
        average_accuracy = self.get_average_accuracy(item_count)

        writer.add_scalar(f'Loss/{mode}', average_loss, epoch)
        writer.add_scalar(f'Accuracy/{mode}', average_accuracy, epoch)

