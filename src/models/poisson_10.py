from src.features.constants import (
    N_SIMS,
    generator_kwargs,
    get_iter_sampling,
    get_iter_warmup,
    model_kwargs,
)
from src.features.poisson_generators import data_generator_poisson_10
from src.features.utils import run_model

if __name__ == "__main__":
    MODEL_NAME = "poisson_10"
    generator = data_generator_poisson_10
    model_kwargs["iter_warmup"] = get_iter_warmup(MODEL_NAME)
    model_kwargs["iter_sampling"] = get_iter_sampling(MODEL_NAME)
    run_model(MODEL_NAME, N_SIMS, generator, generator_kwargs, model_kwargs)
