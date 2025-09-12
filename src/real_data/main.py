# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-positional-arguments, duplicate-code, broad-exception-caught

import logging
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from model_execution import run_real_data_model

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

def run_single_model(args):
    """
    Execute a single model with the provided parameters.

    Args:
        args: Tuple containing (model, season, actual_round, country, num_simulations, ignore_cache)

    Returns:
        str: Execution result or error message
    """
    model, season, actual_round, country, num_simulations, ignore_cache = args

    try:
        run_real_data_model(
            model, season, num_rounds=actual_round, championship=country,
            num_simulations=num_simulations, ignore_cache=ignore_cache
        )
        return f"Success: {model} for {season} with {actual_round} rounds for {country}"
    except Exception:
        return f"Error running {model} for {season} with {actual_round} rounds for {country}"

def process_data(
    model_list: list[str],
    season_list: list[int],
    round_list: list[int | str],
    country_list: list[str],
    num_simulations: int = 100_000,
    ignore_cache: bool = False,
    max_workers: int | None = None,
) -> None:
    """
    Process multiple combinations of models, seasons, rounds, and countries by running
    real data models for each combination in parallel.

    This function iterates through all possible combinations of the provided lists
    and executes the real data model for each combination using parallel processing.

    Args:
        model_list (list): List of model names to process.
        season_list (list): List of seasons (years) to process.
        round_list (list): List of round numbers to process.
        country_list (list): List of country/championship names to process.
        num_simulations (int): Number of simulations to run.
        ignore_cache (bool): Whether to ignore cache.
        max_workers (int, optional): Maximum number of workers. If None, uses cpu_count().

    Note:
        Errors are caught and printed but do not stop the overall process.
    """
    combinations = []
    for values in product(model_list, season_list, round_list, country_list):
        model, season, actual_round, country = values

        if actual_round == "mid" and country in ["france", "germany"]:
            actual_round = 17
        elif actual_round == "mid":
            actual_round = 19
        elif actual_round == "end" and country in ["france", "germany"]:
            actual_round = 34
        elif actual_round == "end":
            actual_round = 38

        combinations.append((model, season, actual_round, country, num_simulations, ignore_cache))

    n_iterations = len(combinations)

    if max_workers is None:
        max_workers = min(cpu_count(), n_iterations)

    print(f"Running {n_iterations} combinations using {max_workers} workers in parallel...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combination = {
            executor.submit(run_single_model, combination): combination
            for combination in combinations
        }

        completed = 0
        for future in as_completed(future_to_combination):
            completed += 1
            result = future.result()

            combination = future_to_combination[future]
            model, season, actual_round, country = combination[:4]

            print(
                f"Progress: {completed}/{n_iterations} - {result} "
                f"{' ' * 30}",
                end="\r",
                flush=True,
            )


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
