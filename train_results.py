from openpyxl import load_workbook

from src.params import Params
from train_random_forest import train_random_forest
from train_generation import train_generation
from train_embedding import train_embedding


# ===== Pre training =====

# Data
data_small = './data/training_data_small.pkl'
data_small_aug = './data/training_data_small_augmented.pkl'

data_medium = './data/training_data_generation.pkl'
data_medium_aug = './data/training_data_generation_augmented.pkl'

data_large = './data/training_data.pkl'
data_large_aug = './data/training_data_augmented.pkl'

# Models
model_path = 'result_model/cbow_network.pt'
model_path_aug = 'result_model/cbow_network_augmented.pt'

# Runs
small_run = {'nb_of_runs': 5, 'column': 'D'}
medium_run = {'nb_of_runs': 3, 'column': 'E'}
large_run = {'nb_of_runs': 1, 'column': 'F'}

# Must do embeddings first.
pretrain_runs = [
    # Embddings 
    {'program': train_embedding, 'run_instance': [large_run], 'row': 67, 'embed': '', 'shuffle': '', 'slide_range': '','generation_data': '', 'embed_data': data_large, 
        'model_path': model_path},
    {'program': train_embedding, 'run_instance': [large_run], 'row': 68, 'embed': '', 'shuffle': '', 'slide_range': '','generation_data': '', 'embed_data': data_large_aug, 
        'model_path': model_path_aug},

    # LSTM Embed
    {'program': train_generation, 'run_instance': [small_run], 'row': 41, 'embed': True, 'shuffle': '', 'slide_range': 1,'generation_data': data_small, 'embed_data': '', 
        'model_path': model_path_aug},
    {'program': train_generation, 'run_instance': [small_run], 'row': 42, 'embed': True, 'shuffle': '', 'slide_range': 1,'generation_data': data_small_aug, 'embed_data': '', 
        'model_path': model_path_aug},
    {'program': train_generation, 'run_instance': [small_run], 'row': 44, 'embed': True, 'shuffle': '', 'slide_range': 5,'generation_data': data_small, 'embed_data': '', 
        'model_path': model_path_aug},
    {'program': train_generation, 'run_instance': [small_run], 'row': 45, 'embed': True, 'shuffle': '', 'slide_range': 5,'generation_data': data_small_aug, 'embed_data': '', 
        'model_path': model_path_aug},

    # LSTM Vanilla
    {'program': train_generation, 'run_instance': [small_run], 'row': 48, 'embed': False, 'shuffle': '', 'slide_range': 1,'generation_data': data_small, 'embed_data': '', 
        'model_path': model_path_aug},
    {'program': train_generation, 'run_instance': [small_run], 'row': 49, 'embed': False, 'shuffle': '', 'slide_range': 1,'generation_data': data_small_aug, 'embed_data': '', 
        'model_path': model_path_aug},
    {'program': train_generation, 'run_instance': [small_run], 'row': 51, 'embed': False, 'shuffle': '', 'slide_range': 5,'generation_data': data_small, 'embed_data': '', 
        'model_path': model_path_aug},
    {'program': train_generation, 'run_instance': [small_run], 'row': 52, 'embed': False, 'shuffle': '', 'slide_range': 5,'generation_data': data_small_aug, 'embed_data': '', 
        'model_path': model_path_aug},

    # Hidden Forest Embed
    {'program': train_random_forest, 'run_instance': [small_run], 'row': 55, 'embed': True, 'shuffle': True, 'slide_range': 1,'generation_data': data_small, 'embed_data': '', 
        'model_path': model_path_aug},
    {'program': train_random_forest, 'run_instance': [small_run], 'row': 56, 'embed': True, 'shuffle': True, 'slide_range': 1,'generation_data': data_small_aug, 'embed_data': '', 
        'model_path': model_path_aug},
    {'program': train_random_forest, 'run_instance': [small_run], 'row': 58, 'embed': True, 'shuffle': True, 'slide_range': 5,'generation_data': data_small, 'embed_data': '', 
        'model_path': model_path_aug},
    {'program': train_random_forest, 'run_instance': [small_run], 'row': 59, 'embed': True, 'shuffle': True, 'slide_range': 5,'generation_data': data_small_aug, 'embed_data': '', 
        'model_path': model_path_aug},

    # Hidden Forest Vanilla
    {'program': train_random_forest, 'run_instance': [small_run], 'row': 62, 'embed': False, 'shuffle': True, 'slide_range': 1,'generation_data': data_small, 'embed_data': '', 
        'model_path': model_path_aug},
    {'program': train_random_forest, 'run_instance': [small_run], 'row': 63, 'embed': False, 'shuffle': True, 'slide_range': 1,'generation_data': data_small_aug, 'embed_data': '', 
        'model_path': model_path_aug},
    {'program': train_random_forest, 'run_instance': [small_run], 'row': 65, 'embed': False, 'shuffle': True, 'slide_range': 5,'generation_data': data_small, 'embed_data': '', 
        'model_path': model_path_aug},
    {'program': train_random_forest, 'run_instance': [small_run], 'row': 66, 'embed': False, 'shuffle': True, 'slide_range': 5,'generation_data': data_small_aug, 'embed_data': '', 
        'model_path': model_path_aug},
    
]

for run in pretrain_runs:
    for run_instance in run['run_instance']:
        results = []

        for run_index in range(run_instance['nb_of_runs']):
            params = Params(embed_data=run['embed'],
                            shuffle_data_random_forest=run['shuffle'],
                            window_slide_range=run['slide_range'],
                            generation_training_data_path=run['generation_data'],
                            embedding_training_data_path=run['embed_data'],
                            embedding_model_path=run['model_path']
                            )
            
            result = run['program'](params)
            results.append(result)

        # Format result. Cannot put '=' at start because excel think it's hacking.
        results = [str(r) for r in results]
        result = ' + '.join(results)
        result = result.replace('.', ',')
        result = f'({result}) / {len(results)} * 100'

        # Write to excel
        workbook = load_workbook(filename = 'results.xlsx')
        sheet = workbook.active

        cell = f"{run_instance['column']}{str(run['row'])}"
        sheet[cell] = result
        workbook.save('results.xlsx')
        print(f'saved cell {cell}')
        
