from os import listdir
import src.song_loader as sl

directory = 'data/'

file_names = listdir(directory)
file_names = file_names[0:18]

for index, file_name in enumerate(file_names):
    print("Data setup file number {} of {}".format(index, len(file_names)))

    song = sl.load_song(directory + file_name)
