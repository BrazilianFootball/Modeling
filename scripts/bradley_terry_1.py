from generators import data_generator_bt_1
from utils import run_model
from constants import (
    generator_kwargs,
    model_kwargs,
    N_SIMS,
    ITER_WARMUP,
    ITER_SAMPLING,
)


if __name__ == "__main__":
    model_name = "bt_model_1"
    generator = data_generator_bt_1
    model_kwargs["iter_warmup"] = ITER_WARMUP[model_name]
    model_kwargs["iter_sampling"] = ITER_SAMPLING[model_name]
    run_model(model_name, N_SIMS, generator, generator_kwargs, model_kwargs)
