N_SIMS = 1_000
N_CLUBS = 20
N_SEASONS = 1

CHAINS = 4
ITER_WARMUP = {
    "bt_model_1": 2_000,
    "bt_model_2": 2_000,
    "poisson_model_1": 2_000,
    "poisson_model_2": 2_000,
    "bad_prior_example": 100,
    "nice_prior_example": 100,
}

ITER_SAMPLING = {
    "bt_model_1": 2_000,
    "bt_model_2": 2_000,
    "poisson_model_1": 2_000,
    "poisson_model_2": 2_000,
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
