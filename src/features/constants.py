N_SIMS = 500
N_CLUBS = 20
N_SEASONS = 1

MODELS = [
    "club_level.bradley_terry_1",
    "club_level.bradley_terry_2",
    "club_level.bradley_terry_3",
    "club_level.bradley_terry_4",
    "club_level.poisson_1",
    "club_level.poisson_2",
    "club_level.poisson_3",
    "club_level.poisson_4",
    "club_level.poisson_5",
    "club_level.poisson_6",
    "club_level.poisson_7",
    "club_level.poisson_8",
    "club_level.poisson_9",
    "club_level.poisson_10",
    # "player_level.bradley_terry_3",
    # "player_level.bradley_terry_4",
    "player_level.poisson_1",
    "player_level.poisson_2",
    "player_level.poisson_3",
    "player_level.poisson_4",
    "player_level.poisson_6",
    "player_level.poisson_7",
    "player_level.poisson_8",
    "player_level.poisson_9",
]

IGNORE_COLS = [
    "chain__",
    "iter__",
    "draw__",
    "lp__",
    "accept_stat__",
    "stepsize__",
    "treedepth__",
    "n_leapfrog__",
    "divergent__",
    "energy__",
    "log_lik",
]

DEFAULT_ITER_WARMUP = 1_000
DEFAULT_ITER_SAMPLING = 1_000

CHAINS = 4
ADAPT_DELTA = 0.95
ADAPT_INIT_PHASE = 150
ADAPT_METRIC_WINDOW = 75
ADAPT_STEP_SIZE = 150
MAX_TREEDEPTH = 10

generator_kwargs = {
    "n_clubs": N_CLUBS,
    "n_seasons": N_SEASONS,
}

model_kwargs = {
    "chains": CHAINS,
    "adapt_delta": ADAPT_DELTA,
    "adapt_init_phase": ADAPT_INIT_PHASE,
    "adapt_metric_window": ADAPT_METRIC_WINDOW,
    "adapt_step_size": ADAPT_STEP_SIZE,
    "max_treedepth": MAX_TREEDEPTH,
}

ITER_WARMUP: dict[str, int] = {
    # "player_level.poisson_1": 1_000,
    # "player_level.poisson_2": 1_000,
    # "player_level.poisson_3": 1_000,
    # "player_level.poisson_4": 1_000,
    # "player_level.poisson_6": 1_000,
    # "player_level.poisson_7": 1_000,
    # "player_level.poisson_8": 1_000,
    # "player_level.poisson_9": 1_000,
}

ITER_SAMPLING: dict[str, int] = {
    # "player_level.poisson_1": 1_000,
    # "player_level.poisson_2": 1_000,
    # "player_level.poisson_3": 1_000,
    # "player_level.poisson_4": 1_000,
    # "player_level.poisson_6": 1_000,
    # "player_level.poisson_7": 1_000,
    # "player_level.poisson_8": 1_000,
    # "player_level.poisson_9": 1_000,
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

    return ITER_SAMPLING.get(model_name, DEFAULT_ITER_SAMPLING)
