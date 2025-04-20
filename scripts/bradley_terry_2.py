from generators import data_generator_single_BT_2
from constants import generator_kwargs, model_kwargs
from utils import run_model

if __name__ == "__main__":
    model_name = "BT_model_2"
    generator = data_generator_single_BT_2
    run_model(model_name, generator, generator_kwargs, model_kwargs)
