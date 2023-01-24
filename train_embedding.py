import random
import torch
from torch.utils.data import DataLoader

import src.midi_generator as midi_generator
import src.note_generator as note_generator
import src.constants as constants

from embedding_data_loader import load_cbow_data, load_skipgram_data
from src.networks.cbow_network import CbowNetwork
from src.networks.skipgram_network import SkipgramNetwork
from src.network_trainer import NetworkTrainer
from src.network_cbow_generator import NetworkCbowGenerator


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

vocabulary, cbow_train_dataset, cbow_valid_dataset, cbow_test_dataset = load_cbow_data()

train_data_loader = DataLoader(cbow_train_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
valid_data_loader = DataLoader(cbow_valid_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
test_data_loader = DataLoader(cbow_test_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)

network = CbowNetwork(len(vocabulary.suffixes_to_indexes.values())).to(device)
trainer = NetworkTrainer(network, train_data_loader, valid_data_loader, test_data_loader)

for epoch in range(1, constants.EPOCHS + 1):
    trainer.epoch_train(epoch)
    trainer.epoch_valid(epoch)

trainer.test()

# Turn off training mode & switch to model evaluation
network.eval()

# === Save model for production use ===
(x_sequence, y_pred) = cbow_train_dataset[0:constants.BATCH_SIZE]
traced_script_module = torch.jit.trace(network.forward, x_sequence.to(device))
traced_script_module.save("result_model/cbow_network.pt")

# Rethink how data is generated. The model is to predict middle word and not the next word. <---------------------------------
# # ==== Code to generate to midi. ====
# random_seeds = random.sample(range(0, len(cbow_train_dataset) - constants.BATCH_SIZE), 9)

# for file_index, song_index in enumerate(random_seeds):
#     print(f'Generating song {file_index + 1}')

#     cbow_generator = NetworkCbowGenerator(network)
#     (x_sequence, y_pred) = cbow_train_dataset[song_index:song_index+constants.BATCH_SIZE]

#     generated_sequence = cbow_generator.generate_sequence(x_sequence)

#     generated_note_infos = note_generator.generate_note_info(generated_sequence, vocabulary)
#     midi_generator.generate_midi(f'generated_file{file_index}.mid', generated_note_infos)
