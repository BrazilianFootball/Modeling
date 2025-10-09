# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-positional-arguments, duplicate-code, broad-exception-caught

from time import time

import logging
from itertools import product

from .model_runner import run_real_data_model

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

def _calculate_total_iterations(model_list: list[str], season_list: list[int],
                               game_list: list[int | str], country_list: list[str]) -> int:
    """Calculate total number of iterations."""
    return len(model_list) * len(season_list) * len(game_list) * len(country_list)

def _should_skip_combination(season: int, country: str) -> bool:
    """Check if combination should be skipped based on business rules."""
    return season == 2019 and country in ["france", "netherlands"]

def _map_game_number(actual_game: int | str, country: str) -> int:
    """Map game identifiers to actual numbers based on country rules."""
    if actual_game == "mid":
        if country in ["france", "germany", "netherlands", "portugal"]:
            return 153
        else:
            return 190
    elif actual_game == "end":
        if country in ["france", "germany", "netherlands", "portugal"]:
            return 306
        else:
            return 380
    else:
        return actual_game

def _format_progress_message(model: str, season: int, actual_game: int, country: str,
                           iteration: int, total_iterations: int, elapsed_time: float) -> str:
    """Format progress message for display."""
    return (
        f"Running {model} for {season} with {actual_game} games for {country} "
        f"(iteration {iteration} of {total_iterations}) "
        f"Time elapsed: {elapsed_time:,.2f} seconds"
        f"{' ' * 30}"
    )

def _execute_single_model(model: str, season: int, actual_game: int, country: str,
                         num_simulations: int, ignore_cache: bool) -> bool:
    """Execute a single model and return success status."""
    try:
        run_real_data_model(
            model, season, num_games=actual_game, championship=country,
            num_simulations=num_simulations, ignore_cache=ignore_cache
        )
        return True
    except Exception as e:
        print(
            f"Error running {model} for {season} with {actual_game} games for {country}: {e}"
            f"{' ' * 40}"
        )
        return False

def _print_final_summary(success: int, total_iterations: int, elapsed_time: float) -> None:
    """Print final execution summary."""
    print(
        f"\nTotal success: {success}/{total_iterations}"
        f"\nTime elapsed: {elapsed_time:,.2f} seconds"
    )

def process_data(
    model_list: list[str],
    season_list: list[int],
    game_list: list[int | str],
    country_list: list[str],
    num_simulations: int = 100_000,
    ignore_cache: bool = False,
) -> None:
    """Process multiple combinations of models, seasons, rounds, and countries."""
    start_time = time()
    success = 0
    total_iterations = _calculate_total_iterations(model_list, season_list, game_list, country_list)

    for i, values in enumerate(product(model_list, season_list, game_list, country_list)):
        model, season, actual_game, country = values

        # Skip invalid combinations
        if _should_skip_combination(season, country):
            continue

        # Map game number
        mapped_game = _map_game_number(actual_game, country)

        # Display progress
        elapsed_time = time() - start_time
        progress_msg = _format_progress_message(
            model, season, mapped_game, country, i+1, total_iterations, elapsed_time
        )
        print(progress_msg, end="\r", flush=True)

        # Execute model
        if _execute_single_model(model, season, mapped_game, country, num_simulations, ignore_cache):
            success += 1

    # Print final summary
    _print_final_summary(success, total_iterations, time() - start_time)


def main() -> None:
    """Main execution function."""
    cmdstanpy_logger = logging.getLogger("cmdstanpy")
    cmdstanpy_logger.disabled = True

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

    games = [50, 100, 150, 200, "mid", "end"]
    countries = ["brazil", "england", "italy", "spain"]
    process_data(models, seasons, games, countries)

    games = [45, 90, 135, 180, "mid", "end"]
    countries = ["france", "germany", "netherlands", "portugal"]
    process_data(models, seasons, games, countries)

if __name__ == "__main__":
    main()
