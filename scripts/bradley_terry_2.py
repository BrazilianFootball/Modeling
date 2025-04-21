from generators import data_generator_bt_2
from constants import generator_kwargs, model_kwargs, N_SIMS
from utils import run_model

if __name__ == "__main__":
    model_name = "bt_model_2"
    generator = data_generator_bt_2
    run_model(model_name, N_SIMS, generator, generator_kwargs, model_kwargs)
