import os
import json
import jsonpickle
import cmdstanpy
import numpy as np

from glob import glob

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def generate_data(generator, n_sims, **kwargs):
    data = dict()
    for i in range(n_sims):
        kwargs['seed'] = i
        data[i] = generator(**kwargs)
    
    return data

def save_model_setup(model_name, setup):
    with open(f'../samples/{model_name}/setup.json', 'w') as f:
        f.write(jsonpickle.encode(setup))

def load_model_setup(model_name):
    with open(f'../samples/{model_name}/setup.json', 'r') as f:
        return jsonpickle.decode(f.read())

def create_model_setup(model_name, data, **kwargs):
    os.makedirs(f'../samples/{model_name}', exist_ok=True)
    setup = {
        'model_name': model_name,
        'data': data
    }

    setup.update(kwargs)

    return setup

def check_changes(current_setup, previous_setup):
    current_dict = json.loads(current_setup)
    previous_dict = json.loads(previous_setup)
    
    for key in current_dict:
        if key not in previous_dict or current_dict[key] != previous_dict[key]:
            return True
            
    for key in previous_dict:
        if key not in current_dict:
            return True

    return False

def setup_was_changed(model_name, data, **kwargs):
    current_setup = create_model_setup(model_name, data, **kwargs)
    if not os.path.exists(f'../samples/{model_name}/setup.json'):
        save_model_setup(model_name, current_setup)
        return True
    else:
        previous_setup = load_model_setup(model_name)
    
    current_json = json.dumps(current_setup, cls=NumpyEncoder, sort_keys=True)
    previous_json = json.dumps(previous_setup, cls=NumpyEncoder, sort_keys=True)
    has_changes = check_changes(current_json, previous_json)
    if has_changes:
        save_model_setup(model_name, current_setup)
    
    return has_changes

def remove_empty_dirs(model_name):
    for root, dirs, _ in os.walk(f'../samples/{model_name}', topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

def clear_samples(model_name):
    for file in glob(f'../samples/{model_name}/**/*'):
        if file.endswith('.csv'):
            os.remove(file)
    
    for file in glob(f'../samples/{model_name}/potential_problems.json'):
        os.remove(file)

    remove_empty_dirs(model_name)    

def run_model(model_name, n_sims, generator, generator_kwargs, model_kwargs):
    kwargs = {**generator_kwargs, **model_kwargs}
    data = generate_data(generator, n_sims, **generator_kwargs)
    need_update = setup_was_changed(model_name, data, **kwargs)

    if need_update:
        clear_samples(model_name)
        potential_problems = dict()
        model = cmdstanpy.CmdStanModel(stan_file=f'../models/{model_name}.stan')
        for i in range(n_sims):
            print(f'Running sim {i+1} of {n_sims} ({model_name})')
            fit = model.sample(data=data[i]['generated'], **model_kwargs)
            fit.save_csvfiles(f'../samples/{model_name}/sim_{i+1}')
            diagnose = fit.diagnose()
            if 'no problems detected' not in diagnose:
                potential_problems[i+1] = diagnose
            
            os.system('clear')

        if potential_problems:
            with open(f'../samples/{model_name}/potential_problems.json', 'w') as f:
                json.dump(potential_problems, f)
            
            print('Potential problems:')
            for i, diagnose in potential_problems.items():
                print(f'Sim {i}:')
                print(diagnose)
                print()
            
            n_potential_problems = len(potential_problems)
            print(f'{n_potential_problems} potential problems detected ({n_potential_problems/n_sims:.2%} of total)')
