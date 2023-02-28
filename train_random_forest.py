from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import src.midi_generator as midi_generator
import src.note_generator as note_generator
import src.constants as constants

from src.generation_data_loader import load_random_forest_data
from src.networks.lstm_embedding_network import LstmEmbeddingNetwork
from src.networks.lstm_vanilla_network import LstmVanillaNetwork
from src.network_trainer import NetworkTrainer
from src.network_sequence_generator import NetworkSequenceGenerator

print('data loading')
vocabulary, X_train, y_train, X_test, y_test = load_random_forest_data()

# train
print('train')

rf_classifier = RandomForestClassifier(n_estimators=100, min_samples_split=8)
rf_classifier.fit(X_train, y_train)

# eval
print('eval')

train_y_pred = rf_classifier.predict(X_train)
train_score = accuracy_score(y_train, train_y_pred)

test_y_pred = rf_classifier.predict(X_test)
val_score = accuracy_score(y_test, test_y_pred)

print(f'train accuracy: {train_score}')
print(f'validation accuracy: {val_score}')

# # === Save model for production use ===
# (_, x_sequence, y_pred) = train_dataset[0:constants.BATCH_SIZE]
# (h, c) = network.get_initial_hidden_context()

# traced_script_module = torch.jit.trace(network.forward, (x_sequence.to(device), (h, c)))
# traced_script_module.save("result_model/generation_network.pt")

# # ==== Code to generate to midi. ====
# random_seeds = random.sample(range(0, len(test_dataset) - constants.BATCH_SIZE), 9)

# for file_index, song_index in enumerate(random_seeds):
#     print(f'Generating song {file_index + 1}')

#     sequence_generator = NetworkSequenceGenerator(network)
#     (_, x_sequence, y_pred) = test_dataset[song_index:song_index+constants.BATCH_SIZE]

#     generated_sequence = sequence_generator.generate_sequence(x_sequence)

#     generated_note_infos = note_generator.generate_note_info(generated_sequence, vocabulary)
#     midi_generator.generate_midi(f'generated_file{file_index}.mid', generated_note_infos)
