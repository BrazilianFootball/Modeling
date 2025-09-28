# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-positional-arguments, duplicate-code, broad-exception-caught

from time import time

import logging
from itertools import product

from model_execution import run_real_data_model

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

def process_data(
    model_list: list[str],
    season_list: list[int],
    game_list: list[int | str],
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
        game_list (list): List of game numbers to process.
        country_list (list): List of country/championship names to process.

    Note:
        Errors are caught and printed but do not stop the overall process.
    """
    start_time = time()
    success = 0
    n_iterations = len(model_list) * len(season_list) * len(game_list) * len(country_list)
    for i, values in enumerate(product(model_list, season_list, game_list, country_list)):
        model, season, actual_game, country = values
        if actual_game == "mid" and country in ["france", "germany", "netherlands", "portugal"]:
            actual_game = 153
        elif actual_game == "mid":
            actual_game = 190
        elif actual_game == "end" and country in ["france", "germany", "netherlands", "portugal"]:
            actual_game = 306
        elif actual_game == "end":
            actual_game = 380

        print(
            f"Running {model} for {season} with {actual_game} games for {country} "
            f"(iteration {i+1} of {n_iterations}) "
            f"Time elapsed: {time() - start_time:,.0f} seconds"
            f"{' ' * 30}",
            end="\r",
            flush=True,
        )
        try:
            run_real_data_model(
                model, season, num_games=actual_game, championship=country,
                num_simulations=num_simulations, ignore_cache=ignore_cache
            )
            success += 1
        except Exception as e:
            print(
                f"Error running {model} for {season} with {actual_game} games for {country}: {e}"
                f"{' ' * 40}"
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
    games: list[int | str] = [50, 100, 150, 200, "mid", "end"]

    # scraped data
    countries = ["brazil"]

    # 20 teams championships
    countries += ["england", "italy", "spain"]

    # 18 teams championships
    countries += ["france", "germany", "netherlands", "portugal"]

    process_data(models, seasons, games, countries)
