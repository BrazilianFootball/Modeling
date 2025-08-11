N_SIMS = 250
N_CLUBS = 20
N_SEASONS = 1

DEFAULT_ITER_WARMUP = 500
DEFAULT_ITER_SAMPLING = 500

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
        "bad_prior_example": 100,
        "nice_prior_example": 100,
    }

    return ITER_SAMPLING.get(model_name, DEFAULT_ITER_SAMPLING)
