N_SIMS = 500
N_CLUBS = 20
N_SEASONS = 1

DEFAULT_ITER_WARMUP = 1000
DEFAULT_ITER_SAMPLING = 1000

CHAINS = 4
ADAPT_DELTA = 0.8
MAX_TREE_DEPTH = 10

generator_kwargs = {
    "n_clubs": N_CLUBS,
    "n_seasons": N_SEASONS,
}

model_kwargs = {
    "chains": CHAINS,
    "adapt_delta": ADAPT_DELTA,
    "max_treedepth": MAX_TREE_DEPTH,
}


def get_iter_warmup(model_name: str) -> int:
    """
    Get the number of warmup iterations for a specific model.

    Args:
        model_name: Name of the model to get warmup iterations for.

    Returns:
        Number of warmup iterations for the specified model, or DEFAULT_ITER_WARMUP
        if the model is not found in the configuration.
    """
    ITER_WARMUP = {
        "bradley_terry_5": 2000,
        "bradley_terry_6": 2000,
        "bradley_terry_7": 2000,
        "bradley_terry_8": 2000,
        "poisson_6": 2000,
        "poisson_7": 2000,
        "poisson_8": 2000,
        "poisson_9": 2000,
        "poisson_10": 2000,
        "bad_prior_example": 100,
        "nice_prior_example": 100,
    }

    return ITER_WARMUP.get(model_name, DEFAULT_ITER_WARMUP)


def get_iter_sampling(model_name: str) -> int:
    """
    Get the number of sampling iterations for a specific model.

    Args:
        model_name: Name of the model to get sampling iterations for.

    Returns:
        Number of sampling iterations for the specified model, or DEFAULT_ITER_SAMPLING
        if the model is not found in the configuration.
    """
    ITER_SAMPLING = {
        "bradley_terry_5": 2000,
        "bradley_terry_6": 2000,
        "bradley_terry_7": 2000,
        "bradley_terry_8": 2000,
        "poisson_6": 2000,
        "poisson_7": 2000,
        "poisson_8": 2000,
        "poisson_9": 2000,
        "poisson_10": 2000,
        "bad_prior_example": 100,
        "nice_prior_example": 100,
    }

    return ITER_SAMPLING.get(model_name, DEFAULT_ITER_SAMPLING)
