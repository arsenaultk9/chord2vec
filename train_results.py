from openpyxl import load_workbook

from src.params import Params
from train_random_forest import train_random_forest
from train_generation import train_generation
from train_embedding import train_embedding


# ===== Pre training =====

small_run = {'nb_of_runs': 5, 'column': 'D'}
medium_run = {'nb_of_runs': 3, 'column': 'E'}
large_run = {'nb_of_runs': 1, 'column': 'F'}

# Must do embeddings first.
pretrain_runs = [
    # Hidden Forest Embed
    {'program': train_random_forest, 'run_instance': [small_run], 'row': 46, 'embed': True, 'shuffle': True, 'slide_range': 5,'generation_data': './data/training_data_small.pkl', 'embed_data': '', 
     'model_path': 'result_model/cbow_network_augmented.pt'},
     {'program': train_random_forest, 'run_instance': [small_run], 'row': 46, 'embed': True, 'shuffle': True, 'slide_range': 5,'generation_data': './data/training_data_small_augmented.pkl', 'embed_data': '', 
     'model_path': 'result_model/cbow_network_augmented.pt'},
]

for run in pretrain_runs:
    for run_instance in run['run_instance']:
        results = []

        for run_index in range(run_instance['nb_of_runs']):
            params = Params(embed_data_random_forest=run['embed'],
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
        
