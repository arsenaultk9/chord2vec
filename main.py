import pickle

with open('./data/training_data.pkl', 'rb') as file:
    data = pickle.load(file, encoding="latin1")

print(data)