import json
 
def load_song(file_path):
    json_file = open(file_path)
    data = json.load(json_file)
    json_file.close()

    # For optimization remove useless parts of song.
    data.pop('InstrumentTracks', None)
    data.pop('Tempos', None)
    data.pop('Scale', None)
    data.pop('SongSections', None)
    data.pop('SubSections', None)

    return data