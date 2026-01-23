# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-positional-arguments, wrong-import-position

import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from data_processing import (
    load_all_matches_data,
    load_real_data,
    mark_cache_modified
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from features.generators import simulate_bradley_terry  # noqa: E402


def get_real_points_evolution(
    data: dict[str, Any], team_mapping: dict[int, str]
) -> dict[str, list[tuple[bool, int, int, int, int]]]:
    """
    Calculate the current points evolution for each team based on real results.

    Args:
        data (Dict[str, Any]): Real data loaded.
        team_mapping (Dict[int, str]): Mapping from team indices to names.

    Returns:
        Dict[str, List[tuple[bool, int, int, int, int]]]: Dictionary with the points evolution
            for each team, including points, wins, goals difference and goals for.
    """
    home_team = data["team1"]
    away_team = data["team2"]
    home_team_names = [team_mapping[team] for team in home_team]
    away_team_names = [team_mapping[team] for team in away_team]

    current_scenario: dict[str, list[tuple[bool, int, int, int, int]]] = {
        team: [] for team in team_mapping.values()
    }
    accumulated_points: dict[str, int] = dict.fromkeys(team_mapping.values(), 0)
    accumulated_wins: dict[str, int] = dict.fromkeys(team_mapping.values(), 0)
    accumulated_goals_diff: dict[str, int] = dict.fromkeys(team_mapping.values(), 0)
    accumulated_goals_for: dict[str, int] = dict.fromkeys(team_mapping.values(), 0)

    for i, (home, away) in enumerate(
        zip(home_team_names, away_team_names, strict=False)
    ):
        goals_home = data["goals_team1"][i]
        goals_away = data["goals_team2"][i]
        if goals_home is not None and goals_away is not None:
            if goals_home > goals_away:
                points_home = 3
                points_away = 0
            elif goals_home < goals_away:
                points_home = 0
                points_away = 3
            else:
                points_home = 1
                points_away = 1
        else:
            points_home = 0
            points_away = 0
            goals_home = 0
            goals_away = 0

        accumulated_points[home] += points_home
        accumulated_wins[home] += int(goals_home > goals_away)
        accumulated_goals_diff[home] += goals_home - goals_away
        accumulated_goals_for[home] += goals_home

        accumulated_points[away] += points_away
        accumulated_wins[away] += int(goals_away > goals_home)
        accumulated_goals_diff[away] += goals_away - goals_home
        accumulated_goals_for[away] += goals_away
        for team in team_mapping.values():
            current_scenario[team].append((
                team in [home, away],
                accumulated_points[team],
                accumulated_wins[team],
                accumulated_goals_diff[team],
                accumulated_goals_for[team]
            ))

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
    # last dimension: (points, wins, goals_diff, goals_for)
    points_matrix = np.zeros(
        (n_teams, num_total_matches - num_games, num_simulations, 4),
        dtype=int
    )
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
                # points
                points_matrix[home_idx, game_simulation_idx, :, 0] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :, 0] + home_new_points
                )
                # wins
                points_matrix[home_idx, game_simulation_idx, :, 1] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :, 1] + (results == 1)
                )
                # goals_diff (supossing all wins are by 1 goal)
                points_matrix[home_idx, game_simulation_idx, :, 2] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :, 2] + (results == 1)
                )
                # goals_for (supossing all wins are by 1 goal)
                points_matrix[home_idx, game_simulation_idx, :, 3] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :, 3] + (results == 1)
                )
            elif team_idx - 1 == away_idx:
                # points
                points_matrix[away_idx, game_simulation_idx, :, 0] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :, 0] + away_new_points
                )
                # wins
                points_matrix[away_idx, game_simulation_idx, :, 1] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :, 1] + (results == 0)
                )
                # goals_diff (supossing all wins are by 1 goal)
                points_matrix[away_idx, game_simulation_idx, :, 2] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :, 2] + (results == 0)
                )
                # goals_for (supossing all wins are by 1 goal)
                points_matrix[away_idx, game_simulation_idx, :, 3] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :, 3] + (results == 0)
                )
            else:
                # repeat previous values
                points_matrix[team_idx - 1, game_simulation_idx, :, :] = (
                    points_matrix[team_idx - 1, game_simulation_idx - 1, :, :]
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

    samples_array = samples.values
    samples_columns_mapping = {col: i for i, col in enumerate(samples.columns)}
    nu_idx = samples_columns_mapping.get("nu", None)

    # last dimension: (points, wins, goals_diff, goals_for)
    points_matrix = np.zeros(
        (n_teams, num_total_matches - num_games, num_simulations, 4),
        dtype=int
    )
    samples_indices = np.random.randint(
        0, len(samples), size=(num_total_matches - num_games, num_simulations)
    )

    probabilities: dict[str, dict[str, Any]] = {}
    for game_id in range(num_games, num_total_matches):
        game_simulation_idx = game_id - num_games
        game_indices = samples_indices[game_simulation_idx]
        if nu_idx is not None:
            nu_values = samples_array[game_indices, nu_idx]
        else:
            nu_values = np.zeros(num_simulations)

        home_name = home_team_names[game_id]
        away_name = away_team_names[game_id]
        if home_name + " (atk home)" in samples.columns:
            home_atk_idx = samples_columns_mapping[home_name + " (atk home)"]
            away_def_idx = samples_columns_mapping[away_name + " (def away)"]
            away_atk_idx = samples_columns_mapping[away_name + " (atk away)"]
            home_def_idx = samples_columns_mapping[home_name + " (def home)"]

            home_atk_strength = samples_array[game_indices, home_atk_idx]
            away_def_strength = samples_array[game_indices, away_def_idx]
            away_atk_strength = samples_array[game_indices, away_atk_idx]
            home_def_strength = samples_array[game_indices, home_def_idx]

            home_strengths = home_atk_strength + away_def_strength
            away_strengths = away_atk_strength + home_def_strength
        elif home_name + " (atk)" in samples.columns:
            home_atk_idx = samples_columns_mapping[home_name + " (atk)"]
            away_def_idx = samples_columns_mapping[away_name + " (def)"]
            away_atk_idx = samples_columns_mapping[away_name + " (atk)"]
            home_def_idx = samples_columns_mapping[home_name + " (def)"]

            home_atk_strength = samples_array[game_indices, home_atk_idx]
            away_def_strength = samples_array[game_indices, away_def_idx]
            away_atk_strength = samples_array[game_indices, away_atk_idx]
            home_def_strength = samples_array[game_indices, home_def_idx]

            home_strengths = home_atk_strength + away_def_strength
            away_strengths = away_atk_strength + home_def_strength
        else:
            home_idx = samples_columns_mapping[home_name]
            away_idx = samples_columns_mapping[away_name]

            home_strengths = samples_array[game_indices, home_idx]
            away_strengths = samples_array[game_indices, away_idx]

        home_strengths = np.exp(home_strengths + nu_values)
        away_strengths = np.exp(away_strengths)

        home_goals = np.random.poisson(home_strengths)
        away_goals = np.random.poisson(away_strengths)

        home_win = home_goals > away_goals
        away_win = home_goals < away_goals
        tie = home_goals == away_goals

        home_idx = home_team[game_id] - 1
        away_idx = away_team[game_id] - 1

        home_new_points = home_win * 3 + tie * 1
        away_new_points = away_win * 3 + tie * 1
        for team_idx in team_mapping:
            if team_idx - 1 == home_idx:
                # points
                points_matrix[home_idx, game_simulation_idx, :, 0] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :, 0] + home_new_points
                )
                # wins
                points_matrix[home_idx, game_simulation_idx, :, 1] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :, 1] + home_win
                )
                # goals_diff
                points_matrix[home_idx, game_simulation_idx, :, 2] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :, 2] + home_goals - away_goals
                )
                # goals_for
                points_matrix[home_idx, game_simulation_idx, :, 3] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :, 3] + home_goals
                )
            elif team_idx - 1 == away_idx:
                # points
                points_matrix[away_idx, game_simulation_idx, :, 0] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :, 0] + away_new_points
                )
                # wins
                points_matrix[away_idx, game_simulation_idx, :, 1] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :, 1] + away_win
                )
                # goals_diff
                points_matrix[away_idx, game_simulation_idx, :, 2] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :, 2] + away_goals - home_goals
                )
                # goals_for
                points_matrix[away_idx, game_simulation_idx, :, 3] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :, 3] + away_goals
                )
            else:
                # repeat previous values
                points_matrix[team_idx - 1, game_simulation_idx, :, :] = (
                    points_matrix[team_idx - 1, game_simulation_idx - 1, :, :]
                )

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


def generate_points_matrix_naive(
    probs: list[float],
    team_mapping: dict[int, str],
    data: dict[str, Any],
    num_games: int,
    num_simulations: int,
    num_total_matches: int,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """
    Simulate the points matrix for the remainder of the season using a naive
    probability-based model.

    This function uses a constant probability vector for the outcomes (home win, draw, away win)
    to simulate each remaining game in the season, and accumulates the points for each team across
    multiple simulation runs. The function returns a matrix of points per team for each round and
    simulation, as well as a dictionary of simulated probabilities for each future game.

    Args:
        probs (list[float]): Probabilities for [home win, tie, away win].
        team_mapping (dict[int, str]): Mapping from team indices to team names.
        data (dict[str, Any]): Dictionary containing real data, with keys "team1" and "team2".
        num_games (int): Number of games already played in the season.
        num_simulations (int): Number of Monte Carlo simulations to run.
        num_total_matches (int): Total number of matches in the season.

    Returns:
        tuple:
            - points_matrix (np.ndarray): Simulated team point totals.
                Shape: (num_teams, num_remaining_games, num_simulations)
            - probabilities (dict[str, dict[str, Any]]): Probabilities for each simulated game,
                keyed by zero-padded game number as a string.
    """
    home_team = data["team1"]
    away_team = data["team2"]
    home_team_names = [team_mapping[team] for team in home_team]
    away_team_names = [team_mapping[team] for team in away_team]
    teams = list(team_mapping.values())
    n_teams = len(teams)
    # last dimension: (points, wins, goals_diff, goals_for)
    points_matrix = np.zeros(
        (n_teams, num_total_matches - num_games, num_simulations, 4),
        dtype=int
    )
    probabilities: dict[str, dict[str, Any]] = {}
    for game_id in range(num_games, num_total_matches):
        game_simulation_idx = game_id - num_games
        results = np.random.choice(
            [1, 0.5, 0], size=(num_simulations), p=probs
        )
        home_idx = home_team[game_id] - 1
        away_idx = away_team[game_id] - 1
        home_new_points = (results == 1) * 3 + (results == 0.5) * 1
        away_new_points = (results == 0) * 3 + (results == 0.5) * 1
        for team_idx in team_mapping:
            if team_idx - 1 == home_idx:
                # points
                points_matrix[home_idx, game_simulation_idx, :, 0] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :, 0] + home_new_points
                )
                # wins
                points_matrix[home_idx, game_simulation_idx, :, 1] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :, 1] + (results == 1)
                )
                # goals_diff (supossing all wins are by 1 goal)
                points_matrix[home_idx, game_simulation_idx, :, 2] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :, 2] + (results == 1)
                )
                # goals_for (supossing all wins are by 1 goal)
                points_matrix[home_idx, game_simulation_idx, :, 3] = (
                    points_matrix[home_idx, game_simulation_idx - 1, :, 3] + (results == 1)
                )
            elif team_idx - 1 == away_idx:
                # points
                points_matrix[away_idx, game_simulation_idx, :, 0] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :, 0] + away_new_points
                )
                # wins
                points_matrix[away_idx, game_simulation_idx, :, 1] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :, 1] + (results == 0)
                )
                # goals_diff (supossing all wins are by 1 goal)
                points_matrix[away_idx, game_simulation_idx, :, 2] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :, 2] + (results == 0)
                )
                # goals_for (supossing all wins are by 1 goal)
                points_matrix[away_idx, game_simulation_idx, :, 3] = (
                    points_matrix[away_idx, game_simulation_idx - 1, :, 3] + (results == 0)
                )
            else:
                # repeat previous values
                points_matrix[team_idx - 1, game_simulation_idx, :, :] = (
                    points_matrix[team_idx - 1, game_simulation_idx - 1, :]
                )

        probabilities[str(game_id + 1).zfill(3)] = {
            "home_team": home_team_names[game_id],
            "away_team": away_team_names[game_id],
            "probabilities": probs,
        }

    return points_matrix, probabilities


def simulate_competition(
    samples: pd.DataFrame,
    team_mapping: dict[int, str],
    model_name: str,
    year: int,
    num_games: int,
    championship: str,
    num_simulations: int = 1_000,
) -> tuple[
        np.ndarray, dict[str, list[tuple[bool, int, int, int, int]]], dict[str, dict[str, Any]]
    ]:
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
        Tuple[
            np.ndarray,
            Dict[str, List[tuple[bool, int, int, int, int]]],
            Dict[str, dict[str, Any]]
        ]:
            points_matrix: Array of simulated points for each team, round, and simulation.
            current_scenario: Dictionary with the actual points evolution for each team.
            probabilities: Dictionary with the probabilities for each simulated game,
                keyed by zero-padded game number as a string.
    """
    data = load_real_data(year, championship)
    n_clubs = len(team_mapping)
    num_total_matches = n_clubs * (n_clubs - 1)

    current_scenario = get_real_points_evolution(data, team_mapping)

    if "naive" in model_name:
        points_matrix, probabilities = generate_points_matrix_naive(
            samples, team_mapping, data, num_games, num_simulations, num_total_matches
        )
    elif "bradley_terry" in model_name:
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
    The data is kept in cache and written to disk only when flush_and_clear_cache()
    is called.

    Args:
        probabilities (dict[str, Any]): The probabilities to update.
        year (int): The year of the data to update.
        model_name (str): The name of the model to update.
        num_games (int): The number of games to update.
        championship (str): The championship of the data.
    """
    data, _ = load_all_matches_data(year, championship)
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

    mark_cache_modified(year, championship)


def calculate_final_positions_probs(
    final_distribution: np.ndarray,
    team_mapping: dict[int, str],
    save_dir: str,
) -> None:
    """
    Calculates the probability of each team finishing in each possible final position.

    For each team, this function computes the probability of finishing in every possible
    position (from first to last) based on the simulated final points distributions.
    The results are saved as a JSON file in the specified directory.

    Args:
        final_distribution (np.ndarray): Array of shape (n_teams, n_simulations, 4)
            containing the simulated final stats for each team across all simulations.
        team_mapping (dict[int, str]): Mapping from team indices (1-based) to team names.
        save_dir (str): Directory where the resulting JSON file will be saved.

    Returns:
        None
    """
    n_teams = len(team_mapping)
    n_simulations = final_distribution.shape[1]
    final_positions = np.zeros((n_teams, n_simulations), dtype=int)
    for sim_idx in range(n_simulations):
        sim_stats = final_distribution[:, sim_idx, :]

        sort_keys = (
            -sim_stats[:, 3],  # Goals for (descending)
            -sim_stats[:, 2],  # Goals diff (descending)
            -sim_stats[:, 1],  # Wins (descending)
            -sim_stats[:, 0],  # Points (descending)
        )
        sorted_indices = np.lexsort(sort_keys)
        final_positions[:, sim_idx] = sorted_indices[::-1]

    final_positions_probs: dict[str, list[float]] = {team: [] for team in team_mapping.values()}
    for idx, team in team_mapping.items():
        for position in range(n_teams):
            prob_team_position = int(np.sum(final_positions[position, :] == idx - 1))
            final_positions_probs[team].insert(0, prob_team_position)

    with open(os.path.join(save_dir, "final_positions_probs.json"), "w", encoding="utf-8") as f:
        json.dump(final_positions_probs, f, ensure_ascii=False, indent=2)
