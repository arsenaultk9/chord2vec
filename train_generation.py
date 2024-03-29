import random
import torch
import torch.optim as optim

from torch.utils.data import DataLoader

import src.midi_generator as midi_generator
import src.note_generator as note_generator
import src.constants as constants

from src.params import get_params, set_params, Params
from src.generation_data_loader import load_generation_data
from src.networks.lstm_embedding_network import LstmEmbeddingNetwork
from src.networks.lstm_vanilla_network import LstmVanillaNetwork
from src.network_trainer import NetworkTrainer
from src.network_sequence_generator import NetworkSequenceGenerator


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def train_generation(params: Params):
    set_params(params)

    vocabulary, train_dataset, valid_dataset, test_dataset = load_generation_data()

    train_data_loader = DataLoader(train_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
    valid_data_loader = DataLoader(valid_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
    test_data_loader = DataLoader(test_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)

    embedding_model = torch.load(get_params().embedding_model_path, map_location=device)
    embedding_weigths = list(embedding_model.parameters())[0]

    network = LstmEmbeddingNetwork(len(vocabulary.suffixes_to_indexes.values()), embedding_weigths).to(device) if get_params(
    ).embed_data else LstmVanillaNetwork(len(vocabulary.suffixes_to_indexes.values()), embedding_weigths).to(device)
    
    trainer = NetworkTrainer(network, train_data_loader, valid_data_loader, test_data_loader, constants.GENERATION_EPOCHS, is_dynamic_lr_scheduler=True)

    for epoch in range(1, constants.GENERATION_EPOCHS + 1):
        trainer.epoch_train(epoch)
        trainer.epoch_valid(epoch)

    test_accuracy = trainer.test()

    # Turn off training mode & switch to model evaluation
    network.eval()

    # === Save model for production use ===
    (_, x_sequence, y_pred) = train_dataset[0:constants.BATCH_SIZE]
    (h, c) = network.get_initial_hidden_context()

    traced_script_module = torch.jit.trace(network.forward, (x_sequence.to(device), (h, c)))
    traced_script_module.save("result_model/generation_network.pt")

    # ==== Code to generate to midi. ====
    random_seeds = random.sample(range(0, len(test_dataset) - constants.BATCH_SIZE), 9)

    for file_index, song_index in enumerate(random_seeds):
        print(f'Generating song {file_index + 1}')

        sequence_generator = NetworkSequenceGenerator(network)
        (_, x_sequence, y_pred) = test_dataset[song_index:song_index+constants.BATCH_SIZE]

        generated_sequence = sequence_generator.generate_sequence(x_sequence)

        generated_note_infos = note_generator.generate_note_info(generated_sequence, vocabulary)
        midi_generator.generate_midi(f'generated_file{file_index}.mid', generated_note_infos)

    return test_accuracy

if __name__ == "__main__":
    train_generation(get_params())