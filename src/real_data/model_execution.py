# pylint: disable=wrong-import-position,too-many-arguments, too-many-positional-arguments, too-many-locals

import json
import os
import shutil
import sys
from typing import Optional
import cmdstanpy
import numpy as np
import pandas as pd
from data_processing import (
    generate_all_matches_data,
    generate_real_data_stan_input,
    check_results_exist,
)
from metrics import calculate_metrics
from simulation import simulate_competition, update_probabilities, calculate_final_positions_probs
from visualization import generate_boxplot, generate_points_evolution_by_team

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from features.constants import model_kwargs, IGNORE_COLS  # noqa: E402


_STAN_MODEL_CACHE: dict[str, cmdstanpy.CmdStanModel] = {}

def get_stan_model(model_name: str) -> cmdstanpy.CmdStanModel:
    """
    Get the Stan model from the cache.
    """
    if model_name not in _STAN_MODEL_CACHE:
        _STAN_MODEL_CACHE[model_name] = cmdstanpy.CmdStanModel(
            stan_file=f"models/club_level/{model_name}.stan",
            force_compile=True
        )
    return _STAN_MODEL_CACHE[model_name]

def run_model_with_real_data(
    model_name: str, year: int, num_games: int = 380, championship: str = "brazil",
    base_path: Optional[str] = None
) -> tuple[cmdstanpy.CmdStanMCMC, dict[int, str], str]:
    """
    Run the specified statistical model (Bradley-Terry or Poisson) using real data
    for a given year and number of rounds. Loads the appropriate data file, prepares
    output directories, compiles the Stan model, and runs sampling. Saves the resulting
    samples to disk.

    Args:
        model_name (str): The name of the model to run ("bradley_terry" or "poisson").
        year (int): The year of the real data to use.
        num_games (int, optional): Number of games to use on fit. Defaults to 380.
        championship (str, optional): The championship of the data. Defaults to "brazil".
        base_path (str, optional): The base path to the real data. Defaults to None.

    Returns:
        Tuple[cmdstanpy.CmdStanMCMC, Dict[int, str], str]:
            fit: The CmdStanPy fit object.
            team_mapping: Dictionary mapping team indices to team names.
            model_name_dir: Directory where model results are saved.
    """
    real_data_file = "bradley_terry" if "bradley_terry" in model_name else "poisson"
    if base_path is None:
        base_path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "real_data", "results", f"{championship}", f"{year}"
        )
        input_path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "real_data", "inputs", f"{championship}", f"{year}",
            f"{real_data_file}_data_{str(num_games).zfill(3)}_games.json"
        )
    else:
        input_path = os.path.join(
            base_path, "inputs",
            f"{real_data_file}_data_{str(num_games).zfill(3)}_games.json"
        )

    save_dir = base_path
    os.makedirs(save_dir, exist_ok=True)
    model_name_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_name_dir, exist_ok=True)

    samples_dir = os.path.join(model_name_dir, f"{str(num_games).zfill(3)}_games")
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)

    os.makedirs(samples_dir)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    team_mapping: dict[int, str] = {
        i + 1: team_name for i, team_name in enumerate(data["team_names"])
    }
    del data["team_names"]
    if "naive_1" in model_name:
        fit = [1/3, 1/3, 1/3]
    elif "naive_2" in model_name:
        home_goals = np.array(data["goals_team1"])
        away_goals = np.array(data["goals_team2"])
        fit = [
            float(np.mean(home_goals > away_goals)),
            float(np.mean(away_goals > home_goals)),
            float(np.mean(home_goals == away_goals)),
        ]
    else:
        stan_model = get_stan_model(model_name)
        fit = stan_model.sample(data=data, show_progress=False, **model_kwargs)
        fit.save_csvfiles(samples_dir)

    return fit, team_mapping, samples_dir


def set_team_strengths(
    samples: pd.DataFrame, team_mapping: dict[int, str]
) -> pd.DataFrame:
    """
    Rename and process the columns of the samples DataFrame to map team indices to team names,
    and compute the overall team strengths depending on the model structure.

    Args:
        samples (pd.DataFrame): DataFrame containing the samples from the model.
        team_mapping (Dict[int, str]): Mapping from team indices to team names.

    Returns:
        pd.DataFrame: DataFrame with columns renamed to team names and team strengths computed.
    """
    n_clubs = len(team_mapping)
    column_mapping: dict[str, str] = {}
    for col in samples.columns:
        if "[" not in col:
            continue
        team_idx = int(col.split("[")[1].split("]")[0])
        if team_idx in team_mapping:
            column_mapping[col] = team_mapping[team_idx]

    if not column_mapping:
        raise ValueError("No skill columns found for teams.")

    if len(column_mapping) == n_clubs:
        samples = samples.rename(columns=column_mapping)
    elif len(column_mapping) == 2 * n_clubs:
        column_mapping = {
            from_value: to_value + " (atk)"
            if "alpha" in from_value
            else to_value + " (def)"
            for from_value, to_value in column_mapping.items()
        }
        samples = samples.rename(columns=column_mapping)
        for team in team_mapping.values():
            samples[team] = samples[team + " (atk)"] - samples[team + " (def)"]
    else:
        map_case = {
            "alpha": " (atk home)",
            "gamma": " (atk away)",
            "delta": " (def home)",
            "beta": " (def away)",
        }
        column_mapping = {
            from_value: to_value + map_case[from_value.split("[")[0]]
            for from_value, to_value in column_mapping.items()
        }
        samples = samples.rename(columns=column_mapping)
        for team in team_mapping.values():
            samples[team] = (
                samples[team + " (atk home)"] + samples[team + " (atk away)"]
            ) / 2 - (samples[team + " (def home)"] - samples[team + " (def away)"]) / 2
    return samples


def run_real_data_model(
    model_name: str,
    year: int,
    num_games: int = 380,
    championship: str = "brazil",
    num_simulations: int = 1_000,
    ignore_cache: bool = False,
    make_plots: bool = False,
    base_path: Optional[str] = None
) -> None:
    """
    Run the specified statistical model (Bradley-Terry or Poisson) using real data
    for a given year and number of rounds, generate a boxplot of team strengths,
    and, if not all rounds are played, simulate the remainder of the season and
    generate a points evolution plot.

    Args:
        model_name (str): The name of the model to run.
        year (int): The year of the real data to use.
        num_games (int, optional): Number of games already played. Defaults to 380.
        championship (str, optional): The championship to use. Defaults to "brazil".
        num_simulations (int, optional): Number of simulations to run. Defaults to 1000.
        ignore_cache (bool, optional): Whether to ignore the cache. Defaults to False.
        make_plots (bool, optional): Whether to make plots. Defaults to False.
        base_path (str, optional): The base path to the real data. Defaults to None.

    Returns:
        None
    """
    if check_results_exist(model_name, year, num_games, championship) and not ignore_cache:
        return

    generate_all_matches_data(year, championship)
    generate_real_data_stan_input(year, num_games, championship)
    fit, team_mapping, model_save_dir = run_model_with_real_data(
        model_name, year, num_games, championship, base_path
    )
    n_clubs = len(team_mapping)
    if "naive" not in model_name:
        samples = fit.draws_pd()
        samples = samples.drop(
            columns=[col for col in samples.columns if "raw" in col] + IGNORE_COLS
        )
        samples = set_team_strengths(samples, team_mapping)
        if make_plots:
            generate_boxplot(
                samples[list(team_mapping.values())],
                year,
                model_save_dir,
                num_games,
            )
    else:
        samples = fit

    if num_games != n_clubs * (n_clubs - 1):
        points_matrix, current_scenario, probabilities = simulate_competition(
            samples, team_mapping, model_name, year, num_games, championship, num_simulations
        )
        update_probabilities(probabilities, year, model_name, num_games, championship)
        final_points_distribution = generate_points_evolution_by_team(
            points_matrix,
            current_scenario,
            team_mapping,
            num_games,
            save_dir=model_save_dir,
            make_plots=make_plots,
        )
        calculate_metrics(model_name, year, num_games, championship)
        calculate_final_positions_probs(
            final_points_distribution,
            team_mapping,
            model_save_dir,
        )
