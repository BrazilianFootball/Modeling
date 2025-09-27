# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-positional-arguments, wrong-import-position

import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from data_processing import load_all_matches_data, load_real_data

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from features.generators import simulate_bradley_terry  # noqa: E402


def get_real_points_evolution(
    data: dict[str, Any], team_mapping: dict[int, str]
) -> dict[str, list[int]]:
    """
    Calculate the current points evolution for each team based on real results.

    Args:
        data (Dict[str, Any]): Real data loaded.
        team_mapping (Dict[int, str]): Mapping from team indices to names.

    Returns:
        Dict[str, List[int]]: Dictionary with the points evolution for each team.
    """
    home_team = data["team1"]
    away_team = data["team2"]
    home_team_names = [team_mapping[team] for team in home_team]
    away_team_names = [team_mapping[team] for team in away_team]

    current_scenario: dict[str, list[int]] = {
        team: [] for team in team_mapping.values()
    }
    accumulated_points: dict[str, int] = dict.fromkeys(team_mapping.values(), 0)

    for i, (home, away) in enumerate(
        zip(home_team_names, away_team_names, strict=False)
    ):
        goals_home = data["goals_team1"][i]
        goals_away = data["goals_team2"][i]
        if goals_home > goals_away:
            points_home = 3
            points_away = 0
        elif goals_home < goals_away:
            points_home = 0
            points_away = 3
        else:
            points_home = 1
            points_away = 1

        accumulated_points[home] += points_home
        accumulated_points[away] += points_away
        for team in team_mapping.values():
            current_scenario[team].append(accumulated_points[team])

    return current_scenario


def generate_points_matrix_bradley_terry(
    samples: pd.DataFrame,
    team_mapping: dict[int, str],
    data: dict[str, Any],
    num_games: int,
    num_simulations: int,
    num_total_matches: int,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """
    Generate a points matrix for the remainder of the season using the Bradley-Terry model.

    Args:
        samples (pd.DataFrame): Posterior samples of team strengths.
        team_mapping (Dict[int, str]): Mapping from team indices to names.
        data (Dict[str, Any]): Real data loaded.
        num_games (int): Number of games already played.
        num_simulations (int): Number of simulations to run.
        num_total_matches (int): Number of matches to simulate.

    Returns:
        np.ndarray: Points matrix of shape (n_teams, n_matches_per_club, num_simulations).
    """
    home_team = data["team1"]
    away_team = data["team2"]
    home_team_names = [team_mapping[team] for team in home_team]
    away_team_names = [team_mapping[team] for team in away_team]
    teams = list(team_mapping.values())
    n_teams = len(teams)
    points_matrix = np.zeros((n_teams, num_total_matches - num_games, num_simulations), dtype=int)
    samples_indices = np.random.randint(
        0, len(samples), size=(num_total_matches - num_games, num_simulations)
    )

    probabilities: dict[str, dict[str, Any]] = {}
    for game_id in range(num_games, num_total_matches):
        game_simulation_idx = game_id - num_games
        home_strengths = samples.iloc[samples_indices[game_simulation_idx]][
            home_team_names[game_id]
        ].values
        away_strengths = samples.iloc[samples_indices[game_simulation_idx]][
            away_team_names[game_id]
        ].values
        if "kappa" in samples.columns:
            kappa_values = samples.iloc[samples_indices[game_simulation_idx]]["kappa"].values
        else:
            kappa_values = np.zeros(num_simulations)

        results = simulate_bradley_terry(
            home_strengths, away_strengths, kappa_values
        )
        home_idx = home_team[game_id] - 1
        away_idx = away_team[game_id] - 1
        home_new_points = (results == 1) * 3 + (results == 0.5) * 1
        away_new_points = (results == 0) * 3 + (results == 0.5) * 1
        for team_idx in team_mapping:
            if team_idx - 1 == home_idx:
                points_matrix[home_idx, game_simulation_idx, :] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :] + home_new_points
                )
            elif team_idx - 1 == away_idx:
                points_matrix[away_idx, game_simulation_idx, :] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :] + away_new_points
                )
            else:
                points_matrix[team_idx - 1, game_simulation_idx, :] = (
                    points_matrix[team_idx - 1, game_simulation_idx - 1, :]
                )

        probs = [
            float(np.sum(results == 1) / num_simulations),
            float(np.sum(results == 0.5) / num_simulations),
            float(np.sum(results == 0) / num_simulations),
        ]
        probabilities[str(game_id + 1).zfill(3)] = {
            "home_team": home_team_names[game_id],
            "away_team": away_team_names[game_id],
            "probabilities": probs,
        }

    return points_matrix, probabilities


def generate_points_matrix_poisson(
    samples: pd.DataFrame,
    team_mapping: dict[int, str],
    data: dict[str, Any],
    num_games: int,
    num_simulations: int,
    num_total_matches: int,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """
    Generate a points matrix for the remainder of the season using the Poisson model.

    Args:
        samples (pd.DataFrame): Posterior samples of team strengths.
        team_mapping (Dict[int, str]): Mapping from team indices to names.
        data (Dict[str, Any]): Real data loaded.
        num_games (int): Number of games already played.
        num_simulations (int): Number of simulations to run.
        num_total_matches (int): Number of matches to simulate.

    Returns:
        np.ndarray: Points matrix of shape (n_teams, n_matches_per_club, num_simulations).
    """
    home_team = data["team1"]
    away_team = data["team2"]
    home_team_names = [team_mapping[team] for team in home_team]
    away_team_names = [team_mapping[team] for team in away_team]
    teams = list(team_mapping.values())
    n_teams = len(teams)
    points_matrix = np.zeros((n_teams, num_total_matches, num_simulations), dtype=int)
    samples_indices = np.random.randint(
        0, len(samples), size=(num_total_matches, num_simulations)
    )

    probabilities: dict[str, dict[str, Any]] = {}
    for game in range(num_total_matches):
        if "nu" in samples.columns:
            nu = samples.iloc[samples_indices[game]]["nu"].values
        else:
            nu = np.zeros(num_simulations)
        for game in range(n_teams // 2):
            game_id = num_games + game
            home_name = home_team_names[game_id]
            away_name = away_team_names[game_id]
            if home_name + " (atk home)" in samples.columns:
                atk_home = samples.iloc[samples_indices[game]][
                    home_name + " (atk home)"
                ].values
                def_away = samples.iloc[samples_indices[game]][
                    away_name + " (def away)"
                ].values
                atk_away = samples.iloc[samples_indices[game]][
                    away_name + " (atk away)"
                ].values
                def_home = samples.iloc[samples_indices[game]][
                    home_name + " (def home)"
                ].values
                home_strengths = atk_home + def_away
                away_strengths = atk_away + def_home
            elif home_name + " (atk)" in samples.columns:
                atk_strength = samples.iloc[samples_indices[game]][
                    home_name + " (atk)"
                ].values
                def_strength = samples.iloc[samples_indices[game]][
                    away_name + " (def)"
                ].values
                home_strengths = atk_strength + def_strength
                away_strengths = atk_strength + def_strength
            else:
                home_strengths = samples.iloc[samples_indices[game]][home_name].values
                away_strengths = samples.iloc[samples_indices[game]][away_name].values

            home_strengths = np.exp(home_strengths + nu)
            away_strengths = np.exp(away_strengths)

            home_goals = np.random.poisson(home_strengths)
            away_goals = np.random.poisson(away_strengths)

            home_win = home_goals > away_goals
            away_win = home_goals < away_goals
            tie = home_goals == away_goals

            home_idx = home_team[game_id] - 1
            away_idx = away_team[game_id] - 1
            if game > 0:
                points_matrix[home_idx, game, :] = (
                    points_matrix[home_idx, game - 1, :] + home_win * 3 + tie * 1
                )
                points_matrix[away_idx, game, :] = (
                    points_matrix[away_idx, game - 1, :] + away_win * 3 + tie * 1
                )
            else:
                points_matrix[home_idx, game, :] = home_win * 3 + tie * 1
                points_matrix[away_idx, game, :] = away_win * 3 + tie * 1

            probs = [
                float(np.sum(home_win) / num_simulations),
                float(np.sum(tie) / num_simulations),
                float(np.sum(away_win) / num_simulations),
            ]
            probabilities[str(game_id + 1).zfill(3)] = {}
            probabilities[str(game_id + 1).zfill(3)]["home_team"] = home_name
            probabilities[str(game_id + 1).zfill(3)]["away_team"] = away_name
            probabilities[str(game_id + 1).zfill(3)]["probabilities"] = probs
    return points_matrix, probabilities


def simulate_competition(
    samples: pd.DataFrame,
    team_mapping: dict[int, str],
    model_name: str,
    year: int,
    num_games: int,
    championship: str,
    num_simulations: int = 1_000,
) -> tuple[np.ndarray, dict[str, list[int]], dict[str, dict[str, Any]]]:
    """
    Simulate the remainder of the Serie A season using posterior samples from the model,
    generating possible points trajectories for each team.

    Args:
        samples (pd.DataFrame): DataFrame with model samples.
        team_mapping (Dict[int, str]): Mapping from team indices to names.
        model_name (str): Model name.
        year (int): Data year.
        num_games (int): Number of games already played.
        championship (str): The championship of the data.
        num_simulations (int, optional): Number of simulations. Default: 1000.

    Returns:
        Tuple[np.ndarray, Dict[str, List[int]]]:
            points_matrix: Array of simulated points for each team, round, and simulation.
            current_scenario: Dictionary with the actual points evolution for each team.
    """
    data = load_real_data(year, championship)
    n_clubs = len(team_mapping)
    num_total_matches = n_clubs * (n_clubs - 1)

    current_scenario = get_real_points_evolution(data, team_mapping)

    if "bradley_terry" in model_name:
        points_matrix, probabilities = generate_points_matrix_bradley_terry(
            samples, team_mapping, data, num_games, num_simulations, num_total_matches
        )
    else:
        points_matrix, probabilities = generate_points_matrix_poisson(
            samples, team_mapping, data, num_games, num_simulations, num_total_matches
        )

    return points_matrix, current_scenario, probabilities


def update_probabilities(
    probabilities: dict[str, Any], year: int, model_name: str, num_games: int, championship: str
) -> None:
    """
    Update the probabilities for a given year, model name, and number of rounds.

    Args:
        probabilities (dict[str, Any]): The probabilities to update.
        year (int): The year of the data to update.
        model_name (str): The name of the model to update.
        num_games (int): The number of games to update.
        championship (str): The championship of the data.
    """
    data, data_path = load_all_matches_data(year, championship)
    for game_id, probabilities_data in probabilities.items():
        assert data[game_id]["home_team"] == probabilities_data["home_team"]
        assert data[game_id]["away_team"] == probabilities_data["away_team"]
        data[game_id]["probabilities"] = data[game_id].get("probabilities", {})
        data[game_id]["probabilities"][model_name] = data[game_id]["probabilities"].get(
            model_name, {}
        )
        data[game_id]["probabilities"][model_name][str(num_games)] = (
            probabilities_data["probabilities"]
        )

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
