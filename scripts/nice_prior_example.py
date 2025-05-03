from constants import ITER_SAMPLING, ITER_WARMUP, model_kwargs
from generators import generate_normal_prior_data
from utils import run_model

if __name__ == "__main__":
    MODEL_NAME = "nice_prior_example"
    N_SIMS = 100

    generator_kwargs = {"n_observations": 5, "true_mu": 0, "true_sigma": 1}
    model_kwargs["iter_warmup"] = ITER_WARMUP[MODEL_NAME]
    model_kwargs["iter_sampling"] = ITER_SAMPLING[MODEL_NAME]
    run_model(
        MODEL_NAME, N_SIMS, generate_normal_prior_data, generator_kwargs, model_kwargs
    )
