# pylint: disable=wrong-import-position

import json
import os
import shutil
import sys

import cmdstanpy
import pandas as pd
from data_processing import generate_all_matches_data, generate_real_data_stan_input
from metrics import calculate_metrics
from simulation import simulate_competition, update_probabilities
from visualization import generate_boxplot, generate_points_evolution_by_team

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from features.constants import model_kwargs, IGNORE_COLS, N_CLUBS  # noqa: E402


def run_model_with_real_data(
    model_name: str, year: int, num_rounds: int = 38
) -> tuple[cmdstanpy.CmdStanMCMC, dict[int, str], str]:
    """
    Run the specified statistical model (Bradley-Terry or Poisson) using real data
    for a given year and number of rounds. Loads the appropriate data file, prepares
    output directories, compiles the Stan model, and runs sampling. Saves the resulting
    samples to disk.

    Args:
        model_name (str): The name of the model to run ("bradley_terry" or "poisson").
        year (int): The year of the real data to use.
        num_rounds (int, optional): Number of rounds to use on fit. Defaults to 38.

    Returns:
        Tuple[cmdstanpy.CmdStanMCMC, Dict[int, str], str]:
            fit: The CmdStanPy fit object.
            team_mapping: Dictionary mapping team indices to team names.
            model_name_dir: Directory where model results are saved.
    """
    save_dir = os.path.join(os.path.dirname(__file__), "../../real_data/results")
    os.makedirs(save_dir, exist_ok=True)
    model_name_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_name_dir, exist_ok=True)

    samples_dir = os.path.join(model_name_dir, f"{year}_{num_rounds}")
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)

    stan_model = cmdstanpy.CmdStanModel(stan_file=f"models/{model_name}.stan")
    real_data_file = "bradley_terry" if "bradley_terry" in model_name else "poisson"
    with open(
        f"real_data/inputs/{real_data_file}_data_{year}_{num_rounds}.json",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    team_mapping: dict[int, str] = {
        i + 1: team_name for i, team_name in enumerate(data["team_names"])
    }
    del data["team_names"]
    fit = stan_model.sample(data=data, **model_kwargs)
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
    column_mapping: dict[str, str] = {}
    for col in samples.columns:
        if "[" not in col:
            continue
        team_idx = int(col.split("[")[1].split("]")[0])
        if team_idx in team_mapping:
            column_mapping[col] = team_mapping[team_idx]

    if not column_mapping:
        raise ValueError("No skill columns found for teams.")

    if len(column_mapping) == N_CLUBS:
        samples = samples.rename(columns=column_mapping)
    elif len(column_mapping) == 2 * N_CLUBS:
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
    model_name: str, year: int, num_rounds: int = 38, num_simulations: int = 1000
) -> None:
    """
    Run the specified statistical model (Bradley-Terry or Poisson) using real data
    for a given year and number of rounds, generate a boxplot of team strengths,
    and, if not all rounds are played, simulate the remainder of the season and
    generate a points evolution plot.

    Args:
        model_name (str): The name of the model to run.
        year (int): The year of the real data to use.
        num_rounds (int, optional): Number of rounds already played. Defaults to 38.
        num_simulations (int, optional): Number of simulations to run. Defaults to 1000.

    Returns:
        None
    """
    generate_all_matches_data(year)
    generate_real_data_stan_input(year, num_rounds)
    fit, team_mapping, model_save_dir = run_model_with_real_data(
        model_name, year, num_rounds
    )
    samples = fit.draws_pd()
    ignore_cols = [col for col in samples.columns if "raw" in col] + IGNORE_COLS
    samples = samples.drop(columns=ignore_cols)
    samples = set_team_strengths(samples, team_mapping)
    generate_boxplot(
        samples[list(team_mapping.values())],
        year,
        model_save_dir,
        num_rounds,
    )
    if num_rounds != 38:
        points_matrix, current_scenario, probabilities = simulate_competition(
            samples, team_mapping, model_name, year, num_rounds, num_simulations
        )
        update_probabilities(probabilities, year, model_name, num_rounds)
        generate_points_evolution_by_team(
            points_matrix,
            current_scenario,
            team_mapping,
            num_rounds,
            save_dir=model_save_dir,
        )
        calculate_metrics(model_name, year, num_rounds)
