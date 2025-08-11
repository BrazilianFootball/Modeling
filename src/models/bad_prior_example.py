from src.features.constants import get_iter_sampling, get_iter_warmup, model_kwargs
from src.features.generators import generate_normal_prior_data
from src.features.utils import run_model

if __name__ == "__main__":
    MODEL_NAME = "bad_prior_example"
    N_SIMS = 100

    generator_kwargs = {"n_observations": 1, "true_mu": 1, "true_sigma": 1}
    model_kwargs["iter_warmup"] = get_iter_warmup(MODEL_NAME)
    model_kwargs["iter_sampling"] = get_iter_sampling(MODEL_NAME)
    run_model(
        MODEL_NAME, N_SIMS, generate_normal_prior_data, generator_kwargs, model_kwargs
    )
