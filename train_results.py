from src.params import Params
from train_random_forest import train_random_forest
from train_generation import train_generation
from train_embedding import train_embedding

params = Params(window_slide_range=5, embedding_training_data_path='./data/training_data_small.pkl')
result = train_embedding(params)

print(f'result: {result}')