import os
import json
import jsonpickle
import cmdstanpy
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def generate_data(generator, **kwargs):
    return generator(**kwargs)

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

def setup_was_changed(model_name, data, **kwargs):
    current_setup = create_model_setup(model_name, data, **kwargs)
    if not os.path.exists(f'../samples/{model_name}/setup.json'):
        save_model_setup(model_name, current_setup)
        return True
    else:
        previous_setup = load_model_setup(model_name)
    
    current_json = json.dumps(current_setup, cls=NumpyEncoder, sort_keys=True)
    previous_json = json.dumps(previous_setup, cls=NumpyEncoder, sort_keys=True)
    if previous_json != current_json:
        save_model_setup(model_name, current_setup)
        return True
    else:
        return False

def clear_samples(model_name):
    for file in os.listdir(f'../samples/{model_name}'):
        if file.endswith('.csv'):
            os.remove(f'../samples/{model_name}/{file}')

def run_model(model_name, generator, generator_kwargs, model_kwargs):
    kwargs = {**generator_kwargs, **model_kwargs}
    data = generate_data(generator, **generator_kwargs)
    need_update = setup_was_changed(model_name, data, **kwargs)

    if need_update:
        clear_samples(model_name)
        model = cmdstanpy.CmdStanModel(stan_file=f'../models/{model_name}.stan')
        fit = model.sample(data=data['generated'], **model_kwargs)
        fit.save_csvfiles(f'../samples/{model_name}')
        
        print("\nDiagnostics:")
        print(fit.diagnose())
