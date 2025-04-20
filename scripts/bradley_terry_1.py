from generators import data_generator_single_BT_1
from constants import generator_kwargs, model_kwargs
from utils import run_model

if __name__ == "__main__":
    model_name = "BT_model_1"
    generator = data_generator_single_BT_1
    run_model(model_name, generator, generator_kwargs, model_kwargs)
