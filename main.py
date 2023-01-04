import random
import torch
from torch.utils.data import DataLoader

import src.midi_generator as midi_generator
import src.note_generator as note_generator
import src.constants as constants

from src.data_loader import load_data
from src.networks.cbow_network import CbowNetwork
from src.network_trainer import NetworkTrainer
from src.network_cbow_generator import NetworkCbowGenerator


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

vocabulary, cbow_dataset = load_data()

train_data_loader = DataLoader(cbow_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)

network = CbowNetwork(len(vocabulary.suffixes_to_indexes.values())).to(device)
trainer = NetworkTrainer(network, train_data_loader)

for epoch in range(1, constants.EPOCHS + 1):
    trainer.epoch_train(epoch)

# Turn off training mode & switch to model evaluation
network.eval()

# ==== Code to generate to midi. ====
random_start_seed = random.randrange(0, len(cbow_dataset) - constants.BATCH_SIZE)

for song_index in range(random_start_seed, random_start_seed + 9):
    file_index = song_index - random_start_seed + 1
    print(f'Generating song {file_index}')

    cbow_generator = NetworkCbowGenerator(network)
    (x_sequence, y_pred) = cbow_dataset[song_index:song_index+constants.BATCH_SIZE]

    generated_sequence = cbow_generator.generate_sequence(x_sequence)

    generated_note_infos = note_generator.generate_note_info(generated_sequence, vocabulary)
    midi_generator.generate_midi(f'generated_file{file_index}.mid', generated_note_infos)
