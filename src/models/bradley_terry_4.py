from src.features.constants import (
    N_SIMS,
    generator_kwargs,
    get_iter_sampling,
    get_iter_warmup,
    model_kwargs,
)
from src.features.generators import data_generator_bt_4
from src.features.utils import run_model

if __name__ == "__main__":
    MODEL_NAME = "bradley_terry_4"
    generator = data_generator_bt_4
    model_kwargs["iter_warmup"] = get_iter_warmup(MODEL_NAME)
    model_kwargs["iter_sampling"] = get_iter_sampling(MODEL_NAME)
    run_model(MODEL_NAME, N_SIMS, generator, generator_kwargs, model_kwargs)
