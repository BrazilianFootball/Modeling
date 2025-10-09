# pylint: disable=wrong-import-position,too-many-arguments, too-many-positional-arguments

import json
import os
import shutil
import sys
from typing import Any

import cmdstanpy
import numpy as np
import pandas as pd
from ..data.data_processing import (
    generate_all_matches_data,
    generate_real_data_stan_input,
    check_results_exist,
)
from ..evaluation.metrics import calculate_metrics
from ..simulation.competition import simulate_competition, update_probabilities, calculate_final_positions_probs
from ..visualization.plots import generate_boxplot, generate_points_evolution_by_team

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from features.constants import model_kwargs, IGNORE_COLS  # noqa: E402


def _create_model_directories(championship: str, year: int, model_name: str, num_games: int) -> tuple[str, str]:
    """Create necessary directories for model execution."""
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "real_data", "results", f"{championship}", f"{year}"
    )
    os.makedirs(save_dir, exist_ok=True)

    model_name_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_name_dir, exist_ok=True)

    samples_dir = os.path.join(model_name_dir, f"{str(num_games).zfill(3)}_games")
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)

    return model_name_dir, samples_dir

def _load_stan_input_data(model_name: str, year: int, num_games: int, championship: str) -> dict[str, Any]:
    """Load Stan input data from JSON file."""
    real_data_file = "bradley_terry" if "bradley_terry" in model_name else "poisson"
    file_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "real_data", "inputs", f"{championship}", f"{year}",
        f"{real_data_file}_data_{str(num_games).zfill(3)}_games.json"
    )

    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

def _create_team_mapping(data: dict[str, Any]) -> dict[int, str]:
    """Create team index to name mapping."""
    return {
        i + 1: team_name for i, team_name in enumerate(data["team_names"])
    }

def _execute_stan_model(model_name: str, data: dict[str, Any], samples_dir: str) -> cmdstanpy.CmdStanMCMC:
    """Execute Stan model and save results."""
    stan_model = cmdstanpy.CmdStanModel(stan_file=f"../models/{model_name}.stan")
    fit = stan_model.sample(data=data, show_progress=False, **model_kwargs)
    fit.save_csvfiles(samples_dir)
    return fit

def run_model_with_real_data(
    model_name: str, year: int, num_games: int = 380, championship: str = "brazil"
) -> tuple[cmdstanpy.CmdStanMCMC, dict[int, str], str]:
    """Run the specified statistical model using real data."""
    # Create directories
    model_name_dir, samples_dir = _create_model_directories(championship, year, model_name, num_games)

    # Load data and create mapping
    data = _load_stan_input_data(model_name, year, num_games, championship)
    team_mapping = _create_team_mapping(data)

    # Remove team names from data
    del data["team_names"]

    # Execute model
    fit = _execute_stan_model(model_name, data, samples_dir)

    return fit, team_mapping, samples_dir


def _extract_team_columns(samples: pd.DataFrame, team_mapping: dict[int, str]) -> dict[str, str]:
    """Extract and map team columns from samples."""
    column_mapping: dict[str, str] = {}
    for col in samples.columns:
        if "[" not in col:
            continue
        team_idx = int(col.split("[")[1].split("]")[0])
        if team_idx in team_mapping:
            column_mapping[col] = team_mapping[team_idx]

    if not column_mapping:
        raise ValueError("No skill columns found for teams.")

    return column_mapping

def _detect_model_type(column_mapping: dict[str, str], n_clubs: int) -> str:
    """Detect the type of model based on number of columns."""
    if len(column_mapping) == n_clubs:
        return "simple"
    elif len(column_mapping) == 2 * n_clubs:
        return "atk_def"
    else:
        return "home_away"

def _apply_simple_model_mapping(samples: pd.DataFrame, column_mapping: dict[str, str]) -> pd.DataFrame:
    """Apply simple model column mapping."""
    return samples.rename(columns=column_mapping)

def _apply_atk_def_model_mapping(samples: pd.DataFrame, column_mapping: dict[str, str],
                                team_mapping: dict[int, str]) -> pd.DataFrame:
    """Apply attack/defense model column mapping and calculate strengths."""
    # Rename columns with (atk) and (def) suffixes
    atk_def_mapping = {
        from_value: to_value + " (atk)" if "alpha" in from_value else to_value + " (def)"
        for from_value, to_value in column_mapping.items()
    }
    samples = samples.rename(columns=atk_def_mapping)

    # Calculate overall team strengths
    for team in team_mapping.values():
        samples[team] = samples[team + " (atk)"] - samples[team + " (def)"]

    return samples

def _apply_home_away_model_mapping(samples: pd.DataFrame, column_mapping: dict[str, str],
                                 team_mapping: dict[int, str]) -> pd.DataFrame:
    """Apply home/away model column mapping and calculate strengths."""
    # Define mapping cases
    map_case = {
        "alpha": " (atk home)",
        "gamma": " (atk away)",
        "delta": " (def home)",
        "beta": " (def away)",
    }

    # Rename columns with home/away suffixes
    home_away_mapping = {
        from_value: to_value + map_case[from_value.split("[")[0]]
        for from_value, to_value in column_mapping.items()
    }
    samples = samples.rename(columns=home_away_mapping)

    # Calculate overall team strengths
    for team in team_mapping.values():
        samples[team] = (
            samples[team + " (atk home)"] + samples[team + " (atk away)"]
        ) / 2 - (samples[team + " (def home)"] - samples[team + " (def away)"]) / 2

    return samples

def set_team_strengths(samples: pd.DataFrame, team_mapping: dict[int, str]) -> pd.DataFrame:
    """Rename columns and compute team strengths based on model structure."""
    # Extract team columns
    column_mapping = _extract_team_columns(samples, team_mapping)
    n_clubs = len(team_mapping)

    # Detect model type and apply appropriate mapping
    model_type = _detect_model_type(column_mapping, n_clubs)

    if model_type == "simple":
        return _apply_simple_model_mapping(samples, column_mapping)
    elif model_type == "atk_def":
        return _apply_atk_def_model_mapping(samples, column_mapping, team_mapping)
    else:  # home_away
        return _apply_home_away_model_mapping(samples, column_mapping, team_mapping)


def _check_cache_and_return(model_name: str, year: int, num_games: int,
                           championship: str, ignore_cache: bool) -> bool:
    """Check if results exist in cache and return early if found."""
    if check_results_exist(model_name, year, num_games, championship) and not ignore_cache:
        return True
    return False

def _prepare_data_for_model(year: int, num_games: int, championship: str) -> None:
    """Prepare all necessary data for model execution."""
    generate_all_matches_data(year, championship)
    generate_real_data_stan_input(year, num_games, championship)

def _execute_model_pipeline(model_name: str, year: int, num_games: int,
                          championship: str) -> tuple[cmdstanpy.CmdStanMCMC, dict[int, str], str]:
    """Execute the complete model pipeline."""
    fit, team_mapping, model_save_dir = run_model_with_real_data(
        model_name, year, num_games, championship
    )
    return fit, team_mapping, model_save_dir

def _process_model_samples(fit: cmdstanpy.CmdStanMCMC, team_mapping: dict[int, str]) -> pd.DataFrame:
    """Process model samples and calculate team strengths."""
    samples = fit.draws_pd()
    samples = samples.drop(
        columns=[col for col in samples.columns if "raw" in col] + IGNORE_COLS
    )
    return set_team_strengths(samples, team_mapping)

def _generate_team_visualizations(samples: pd.DataFrame, team_mapping: dict[int, str],
                                year: int, model_save_dir: str, num_games: int) -> None:
    """Generate team strength visualizations."""
    generate_boxplot(
        samples[list(team_mapping.values())],
        year,
        model_save_dir,
        num_games,
    )

def _should_simulate_competition(num_games: int, n_clubs: int) -> bool:
    """Determine if competition simulation is needed."""
    return num_games != n_clubs * (n_clubs - 1)

def _run_simulation_pipeline(samples: pd.DataFrame, team_mapping: dict[int, str],
                           model_name: str, year: int, num_games: int,
                           championship: str, num_simulations: int) -> tuple[np.ndarray, dict, dict]:
    """Run the complete simulation pipeline."""
    points_matrix, current_scenario, probabilities = simulate_competition(
        samples, team_mapping, model_name, year, num_games, championship, num_simulations
    )
    return points_matrix, current_scenario, probabilities

def _update_simulation_data(probabilities: dict, year: int, model_name: str,
                           num_games: int, championship: str) -> None:
    """Update simulation data with new probabilities."""
    update_probabilities(probabilities, year, model_name, num_games, championship)

def _generate_simulation_visualizations(points_matrix: np.ndarray, current_scenario: dict,
                                       team_mapping: dict[int, str], num_games: int,
                                       model_save_dir: str) -> np.ndarray:
    """Generate simulation visualizations."""
    return generate_points_evolution_by_team(
        points_matrix,
        current_scenario,
        team_mapping,
        num_games,
        save_dir=model_save_dir,
    )

def _calculate_final_metrics_and_positions(model_name: str, year: int, num_games: int,
                                         championship: str, final_points_distribution: np.ndarray,
                                         team_mapping: dict[int, str], model_save_dir: str) -> None:
    """Calculate final metrics and position probabilities."""
    calculate_metrics(model_name, year, num_games, championship)
    calculate_final_positions_probs(
        final_points_distribution,
        team_mapping,
        model_save_dir,
    )

def run_real_data_model(
    model_name: str,
    year: int,
    num_games: int = 380,
    championship: str = "brazil",
    num_simulations: int = 1_000,
    ignore_cache: bool = False,
) -> None:
    """Run the complete real data model pipeline."""
    # Check cache and return early if results exist
    if _check_cache_and_return(model_name, year, num_games, championship, ignore_cache):
        return

    # Prepare data
    _prepare_data_for_model(year, num_games, championship)

    # Execute model
    fit, team_mapping, model_save_dir = _execute_model_pipeline(
        model_name, year, num_games, championship
    )

    # Process samples
    samples = _process_model_samples(fit, team_mapping)

    # Generate team visualizations
    _generate_team_visualizations(samples, team_mapping, year, model_save_dir, num_games)

    # Check if simulation is needed
    n_clubs = len(team_mapping)
    if _should_simulate_competition(num_games, n_clubs):
        # Run simulation pipeline
        points_matrix, current_scenario, probabilities = _run_simulation_pipeline(
            samples, team_mapping, model_name, year, num_games, championship, num_simulations
        )

        # Update simulation data
        _update_simulation_data(probabilities, year, model_name, num_games, championship)

        # Generate simulation visualizations
        final_points_distribution = _generate_simulation_visualizations(
            points_matrix, current_scenario, team_mapping, num_games, model_save_dir
        )

        # Calculate final metrics and positions
        _calculate_final_metrics_and_positions(
            model_name, year, num_games, championship, final_points_distribution,
            team_mapping, model_save_dir
        )
