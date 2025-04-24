N_SIMS = 200
N_CLUBS = 20
N_SEASONS = 1

CHAINS = 4
ITER_WARMUP = {
    "bt_model_1": 1_000,
    "bt_model_2": 1_000,
    "poisson_model_1": 1_000,
    "poisson_model_2": 1_000,
}

ITER_SAMPLING = {
    "bt_model_1": 1_000,
    "bt_model_2": 1_000,
    "poisson_model_1": 1_000,
    "poisson_model_2": 1_000,
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
