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
    rounds = [5, 10, 15, 19, 20, 38]
    countries = ["brazil", "england", "france", "germany", "italy", "spain"]

    for i, values in enumerate(product(models, seasons, rounds, countries)):
        model, season, actual_round, country = values
        print(
            f"Running {model} for {season} with {actual_round} rounds for {country} "
            f"(iteration {i+1} of {len(models)*len(seasons)*len(rounds)*len(countries)})",
            end="\r",
        )
        run_real_data_model(
            model, season, num_rounds=actual_round, championship=country, num_simulations=100_000
        )
