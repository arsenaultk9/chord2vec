import pickle
from os import listdir

import src.song_loader as sl
import src.chords_extractor as ce
from src.models.chords_vocab_builder import build_chords_vocab
from src.training_data_preparator import get_training_data


directory = 'C:/dev/data/KafkaSongs-mer., janv. 18, 2023/'

file_names = listdir(directory)
file_names = file_names #[0:18]

all_song_chords = []

for index, file_name in enumerate(file_names):
    print("Data setup file number {} of {}".format(index, len(file_names)))

    try:
        song = sl.load_song(directory + file_name)
        song_chords = ce.extract_chords(song)

        all_song_chords.append(song_chords)
    except Exception as e:
        print(e)

chords_vocabulary = build_chords_vocab()
all_data = get_training_data(all_song_chords, chords_vocabulary)

all_data_lenght = len(all_data)
test_split = all_data_lenght - int((all_data_lenght * 0.10))
valid_split = test_split - int((all_data_lenght * 0.20))

train_data = all_data[0:valid_split]
valid_data = all_data[valid_split:test_split]
test_data = all_data[test_split:all_data_lenght]

data = {
    "chords_vocabulary": chords_vocabulary,
    "train": train_data,
    "valid": valid_data,
    "test": test_data
}

# Store data (serialize)
with open('./data/training_data.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)