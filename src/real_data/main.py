# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-positional-arguments, duplicate-code

import logging
from itertools import product

from model_execution import run_real_data_model

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

if __name__ == "__main__":
    models = [
        "bradley_terry_3",
        "bradley_terry_4",
        "poisson_1",
        "poisson_2",
        "poisson_3",
        "poisson_4",
        "poisson_5",
    ]

    seasons = [*range(2019, 2025)]
    rounds = [5, 10, 15, 19, 20]
    models = [
        "bradley_terry_3",
    ]
    seasons = [2024]
    rounds = [19, 20]

    for model, season, actual_round in product(models, seasons, rounds):
        run_real_data_model(
            model, season, num_rounds=actual_round, num_simulations=100_000
        )
