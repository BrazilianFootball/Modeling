import os
from typing import Tuple

def get_real_data_root() -> str:
    """Get the real_data root directory."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "real_data")

def get_results_path(championship: str, year: int) -> str:
    """Get results directory path."""
    return os.path.join(get_real_data_root(), "results", f"{championship}", f"{year}")

def get_inputs_path(championship: str, year: int) -> str:
    """Get inputs directory path."""
    return os.path.join(get_real_data_root(), "inputs", f"{championship}", f"{year}")

def get_model_results_path(championship: str, year: int, model_name: str, num_games: int) -> str:
    """Get model results directory path."""
    return os.path.join(
        get_results_path(championship, year),
        model_name,
        f"{str(num_games).zfill(3)}_games"
    )

def get_stan_input_file_path(championship: str, year: int, model_type: str, num_games: int) -> str:
    """Get Stan input file path."""
    return os.path.join(
        get_inputs_path(championship, year),
        f"{model_type}_data_{str(num_games).zfill(3)}_games.json"
    )
