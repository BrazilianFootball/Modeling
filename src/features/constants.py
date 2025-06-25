N_SIMS = 250
N_CLUBS = 20
N_SEASONS = 1

DEFAULT_ITER_WARMUP = 500
DEFAULT_ITER_SAMPLING = 500

CHAINS = 4
ITER_WARMUP = {
    "bt_model_1": DEFAULT_ITER_WARMUP,
    "bt_model_2": DEFAULT_ITER_WARMUP,
    "poisson_model_1": DEFAULT_ITER_WARMUP,
    "poisson_model_2": DEFAULT_ITER_WARMUP,
    "karlis_ntzoufras_1": DEFAULT_ITER_WARMUP,
    "karlis_ntzoufras_2": DEFAULT_ITER_WARMUP,
    "bad_prior_example": 100,
    "nice_prior_example": 100,
}

ITER_SAMPLING = {
    "bt_model_1": DEFAULT_ITER_SAMPLING,
    "bt_model_2": DEFAULT_ITER_SAMPLING,
    "poisson_model_1": DEFAULT_ITER_SAMPLING,
    "poisson_model_2": DEFAULT_ITER_SAMPLING,
    "karlis_ntzoufras_1": DEFAULT_ITER_SAMPLING,
    "karlis_ntzoufras_2": DEFAULT_ITER_SAMPLING,
    "bad_prior_example": 100,
    "nice_prior_example": 100,
}

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
