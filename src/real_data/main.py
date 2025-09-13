# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-positional-arguments, duplicate-code, broad-exception-caught

import logging
from itertools import product

from model_execution import run_real_data_model

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

def process_data(
    model_list: list[str],
    season_list: list[int],
    round_list: list[int | str],
    country_list: list[str],
    num_simulations: int = 100_000,
    ignore_cache: bool = False,
) -> None:
    """
    Process multiple combinations of models, seasons, rounds, and countries by running
    real data models for each combination.

    This function iterates through all possible combinations of the provided lists
    and executes the real data model for each combination. It provides progress
    feedback and handles errors gracefully.

    Args:
        model_list (list): List of model names to process.
        season_list (list): List of seasons (years) to process.
        round_list (list): List of round numbers to process.
        country_list (list): List of country/championship names to process.

    Note:
        Errors are caught and printed but do not stop the overall process.
    """
    success = 0
    n_iterations = len(model_list) * len(season_list) * len(round_list) * len(country_list)
    for i, values in enumerate(product(model_list, season_list, round_list, country_list)):
        model, season, actual_round, country = values
        if actual_round == "mid" and country in ["france", "germany"]:
            actual_round = 17
        elif actual_round == "mid":
            actual_round = 19
        elif actual_round == "end" and country in ["france", "germany"]:
            actual_round = 34
        elif actual_round == "end":
            actual_round = 38

        print(
            f"Running {model} for {season} with {actual_round} rounds for {country} "
            f"(iteration {i+1} of {n_iterations})"
            f"{' ' * 30}",
            end="\r",
            flush=True,
        )
        try:
            run_real_data_model(
                model, season, num_rounds=actual_round, championship=country,
                num_simulations=num_simulations, ignore_cache=ignore_cache
            )
            success += 1
        except Exception as e:
            print(
                f"Error running {model} for {season} with {actual_round} rounds for {country}: {e}"
                f"{' ' * 30}"
            )

    print(f"\nTotal success: {success}/{n_iterations}")


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
    rounds: list[int | str] = [5, 10, 15, 20, "mid", "end"]
    countries = ["brazil"]
    process_data(models, seasons, rounds, countries)

    seasons = [2023, 2024]
    countries = ["england", "france", "germany", "italy", "spain"]
    process_data(models, seasons, rounds, countries)
