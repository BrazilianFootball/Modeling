N_SIMS = 20
N_CLUBS = 20
N_SEASONS = 1

CHAINS = 4
ITER_WARMUP = 100
ITER_SAMPLING = 100

ADAPT_DELTA = 0.8
MAX_TREE_DEPTH = 10

generator_kwargs = {
    'n_clubs': N_CLUBS,
    'n_seasons': N_SEASONS,
}

model_kwargs = {
    'chains': CHAINS,
    'iter_warmup': ITER_WARMUP,
    'iter_sampling': ITER_SAMPLING,
    'adapt_delta': ADAPT_DELTA,
    'max_treedepth': MAX_TREE_DEPTH,
}
