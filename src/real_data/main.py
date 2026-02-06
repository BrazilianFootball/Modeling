# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-positional-arguments, duplicate-code, broad-exception-caught

from time import time
import logging
from itertools import product

from tqdm import tqdm

from data_processing import flush_and_clear_cache
from metrics import flush_metrics_cache
from model_execution import run_real_data_model

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

def _process_data(
    model_list: list[str],
    season_list: list[int],
    game_list: list[int],
    game_to_plot_list: list[int],
    country_name: str,
    num_simulations: int = 10_000,
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
        game_to_plot_list (list): List of game numbers to plot.
        country_name (str): Country/championship name to process.
        num_simulations (int): Number of simulations to run.
        ignore_cache (bool): Whether to ignore the cache.

    Note:
        Errors are caught and printed but do not stop the overall process.
    """
    start_time = time()
    success = 0
    n_iterations = len(model_list) * len(season_list) * len(game_list)
    for values in tqdm(product(model_list, season_list, game_list), total=n_iterations):
        model, season, actual_game = values
        try:
            run_real_data_model(
                model, season, num_games=actual_game, championship=country_name,
                num_simulations=num_simulations, ignore_cache=ignore_cache,
                make_plots=actual_game in game_to_plot_list
            )
            success += 1
        except Exception as e:
            print(
                f"Error running {model} for {season} with {actual_game} games for {country_name}:"
                f"\n{e}"
            )
            raise e

    print(
        f"\nTotal success: {success}/{n_iterations}"
        f"\nTime elapsed: {time() - start_time:,.2f} seconds"
    )

def process_data(
    model_list: list[str],
    season_list: list[int],
    game_list: list[int],
    game_to_plot_list: list[int],
    country_name: str,
    num_simulations: int = 10_000,
    ignore_cache: bool = False,
) -> None:
    """
    Process the data for a given model, season, game, and country.
    """
    try:
        _process_data(
            model_list,
            season_list,
            game_list,
            game_to_plot_list,
            country_name,
            num_simulations,
            ignore_cache
        )
    finally:
        flush_and_clear_cache()
        flush_metrics_cache()


if __name__ == "__main__":
    models = [
        "naive_1",
        "naive_2",
        "bradley_terry_3",
        "bradley_terry_4",
        "poisson_1",
        "poisson_2",
        "poisson_3",
        "poisson_4",
        "poisson_5",
        "poisson_6",
        "poisson_7",
        "poisson_8",
        "poisson_9",
        "poisson_10",
    ]

    # === 20 teams championships ===
    games_20 = [*range(50, 381, 10)]
    games_to_plot_20 = [50, 100, 150, 200, 190, 380]

    # Brazil (2019-2025)
    process_data(models, [*range(2019, 2026)], games_20, games_to_plot_20, "brazil")

    # England (2019-2024)
    process_data(models, [*range(2019, 2025)], games_20, games_to_plot_20, "england")

    # Italy (2019-2024)
    process_data(models, [*range(2019, 2025)], games_20, games_to_plot_20, "italy")

    # Spain (2019-2024)
    process_data(models, [*range(2019, 2025)], games_20, games_to_plot_20, "spain")

    # === 18 teams championships ===
    games_18 = [*range(45, 307, 9)]
    games_to_plot_18 = [45, 90, 135, 180, 153, 306]

    # France (2020-2022 with 20 teams)
    process_data(models, [*range(2020, 2023)], games_20, games_to_plot_20, "france")

    # France (2023-2024 with 18 teams)
    process_data(models, [*range(2023, 2025)], games_18, games_to_plot_18, "france")

    # Germany (2019-2024)
    process_data(models, [*range(2019, 2025)], games_18, games_to_plot_18, "germany")

    # Netherlands (2020-2024, skips 2019)
    process_data(models, [*range(2020, 2025)], games_18, games_to_plot_18, "netherlands")

    # Portugal (2019-2024)
    process_data(models, [*range(2019, 2025)], games_18, games_to_plot_18, "portugal")
