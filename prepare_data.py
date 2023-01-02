import pickle
from os import listdir

import src.song_loader as sl
import src.chords_extractor as ce
from src.models.chords_vocab_builder import build_chords_vocab


directory = 'data/'

file_names = listdir(directory)
file_names = file_names[0:18]

chords_vocabulary = build_chords_vocab()
all_song_chords = []
training_data = []

for index, file_name in enumerate(file_names):
    print("Data setup file number {} of {}".format(index, len(file_names)))

    song = sl.load_song(directory + file_name)
    song_chords = ce.extract_chords(song)

    all_song_chords.append(song_chords)

data = {
    "chords_vocabulary": chords_vocabulary,
    "data": training_data
}

# Store data (serialize)
with open('./data/training_data.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)