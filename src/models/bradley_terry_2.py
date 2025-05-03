from src.features.constants import (
    ITER_SAMPLING,
    ITER_WARMUP,
    N_SIMS,
    generator_kwargs,
    model_kwargs,
)
from src.features.generators import data_generator_bt_2
from src.features.utils import run_model

if __name__ == "__main__":
    MODEL_NAME = "bt_model_2"
    generator = data_generator_bt_2
    model_kwargs["iter_warmup"] = ITER_WARMUP[MODEL_NAME]
    model_kwargs["iter_sampling"] = ITER_SAMPLING[MODEL_NAME]
    run_model(MODEL_NAME, N_SIMS, generator, generator_kwargs, model_kwargs)
