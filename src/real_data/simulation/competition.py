# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-positional-arguments, wrong-import-position

import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from ..data.data_processing import load_all_matches_data, load_real_data
from ..utils.io_utils import save_json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from features.generators import simulate_bradley_terry  # noqa: E402


def get_real_points_evolution(
    data: dict[str, Any], team_mapping: dict[int, str]
) -> dict[str, list[tuple[bool, int]]]:
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

    current_scenario: dict[str, list[tuple[bool, int]]] = {
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
            current_scenario[team].append((team in [home, away], accumulated_points[team]))

    return current_scenario


def _extract_team_data_for_simulation(data: dict[str, Any], team_mapping: dict[int, str]) -> tuple:
    """Extract team data needed for simulation."""
    home_team = data["team1"]
    away_team = data["team2"]
    home_team_names = [team_mapping[team] for team in home_team]
    away_team_names = [team_mapping[team] for team in away_team]
    return home_team, away_team, home_team_names, away_team_names

def _generate_random_sample_indices(num_games: int, num_simulations: int, samples_length: int) -> np.ndarray:
    """Generate random indices for sampling."""
    return np.random.randint(0, samples_length, size=(num_games, num_simulations))

def _extract_team_strengths_bradley_terry(samples: pd.DataFrame, home_team_names: list[str],
                                        away_team_names: list[str], game_id: int,
                                        sample_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract team strengths for Bradley-Terry simulation."""
    home_strengths = samples.iloc[sample_indices][home_team_names[game_id]].values
    away_strengths = samples.iloc[sample_indices][away_team_names[game_id]].values

    if "kappa" in samples.columns:
        kappa_values = samples.iloc[sample_indices]["kappa"].values
    else:
        kappa_values = np.zeros(len(sample_indices))

    return home_strengths, away_strengths, kappa_values

def _simulate_game_results_bradley_terry(home_strengths: np.ndarray, away_strengths: np.ndarray,
                                       kappa_values: np.ndarray) -> np.ndarray:
    """Simulate game results using Bradley-Terry model."""
    return simulate_bradley_terry(home_strengths, away_strengths, kappa_values)

def _calculate_points_from_results(results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate points for home and away teams from game results."""
    home_new_points = (results == 1) * 3 + (results == 0.5) * 1
    away_new_points = (results == 0) * 3 + (results == 0.5) * 1
    return home_new_points, away_new_points

def _update_points_matrix(points_matrix: np.ndarray, home_idx: int, away_idx: int,
                         game_simulation_idx: int, home_new_points: np.ndarray,
                         away_new_points: np.ndarray, team_mapping: dict[int, str]) -> None:
    """Update points matrix with new points for current game."""
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

def _calculate_game_probabilities(results: np.ndarray, num_simulations: int) -> list[float]:
    """Calculate probabilities for game outcomes."""
    return [
        float(np.sum(results == 1) / num_simulations),
        float(np.sum(results == 0.5) / num_simulations),
        float(np.sum(results == 0) / num_simulations),
    ]

def _create_probability_entry(home_team_name: str, away_team_name: str,
                            probabilities: list[float], game_id: int) -> dict[str, Any]:
    """Create probability entry for a game."""
    return {
        "home_team": home_team_name,
        "away_team": away_team_name,
        "probabilities": probabilities,
    }

def generate_points_matrix_bradley_terry(
    samples: pd.DataFrame,
    team_mapping: dict[int, str],
    data: dict[str, Any],
    num_games: int,
    num_simulations: int,
    num_total_matches: int,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """Generate a points matrix for the remainder of the season using the Bradley-Terry model."""
    # Extract data
    home_team, away_team, home_team_names, away_team_names = _extract_team_data_for_simulation(data, team_mapping)
    n_teams = len(team_mapping)

    # Initialize matrices
    points_matrix = np.zeros((n_teams, num_total_matches - num_games, num_simulations), dtype=int)
    samples_indices = _generate_random_sample_indices(num_total_matches - num_games, num_simulations, len(samples))
    probabilities: dict[str, dict[str, Any]] = {}

    # Simulate each game
    for game_id in range(num_games, num_total_matches):
        game_simulation_idx = game_id - num_games

        # Extract strengths and simulate
        home_strengths, away_strengths, kappa_values = _extract_team_strengths_bradley_terry(
            samples, home_team_names, away_team_names, game_id, samples_indices[game_simulation_idx]
        )

        results = _simulate_game_results_bradley_terry(home_strengths, away_strengths, kappa_values)

        # Calculate points and update matrix
        home_new_points, away_new_points = _calculate_points_from_results(results)
        home_idx = home_team[game_id] - 1
        away_idx = away_team[game_id] - 1

        _update_points_matrix(points_matrix, home_idx, away_idx, game_simulation_idx,
                            home_new_points, away_new_points, team_mapping)

        # Calculate probabilities
        probs = _calculate_game_probabilities(results, num_simulations)
        probabilities[str(game_id + 1).zfill(3)] = _create_probability_entry(
            home_team_names[game_id], away_team_names[game_id], probs, game_id
        )

    return points_matrix, probabilities


def _detect_poisson_model_type(samples: pd.DataFrame, home_name: str, away_name: str) -> str:
    """Detect the type of Poisson model based on column names."""
    if home_name + " (atk home)" in samples.columns:
        return "home_away"
    elif home_name + " (atk)" in samples.columns:
        return "atk_def"
    else:
        return "simple"

def _extract_poisson_strengths_home_away(samples_array: np.ndarray, samples_columns_mapping: dict[str, int],
                                       home_name: str, away_name: str, game_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract strengths for home/away Poisson model."""
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

    return home_strengths, away_strengths

def _extract_poisson_strengths_atk_def(samples_array: np.ndarray, samples_columns_mapping: dict[str, int],
                                     home_name: str, away_name: str, game_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract strengths for atk/def Poisson model."""
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

    return home_strengths, away_strengths

def _extract_poisson_strengths_simple(samples_array: np.ndarray, samples_columns_mapping: dict[str, int],
                                    home_name: str, away_name: str, game_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract strengths for simple Poisson model."""
    home_idx = samples_columns_mapping[home_name]
    away_idx = samples_columns_mapping[away_name]

    home_strengths = samples_array[game_indices, home_idx]
    away_strengths = samples_array[game_indices, away_idx]

    return home_strengths, away_strengths

def _extract_poisson_strengths(samples: pd.DataFrame, home_name: str, away_name: str,
                              game_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract team strengths based on Poisson model type."""
    samples_array = samples.values
    samples_columns_mapping = {col: i for i, col in enumerate(samples.columns)}

    model_type = _detect_poisson_model_type(samples, home_name, away_name)

    if model_type == "home_away":
        return _extract_poisson_strengths_home_away(samples_array, samples_columns_mapping,
                                                  home_name, away_name, game_indices)
    elif model_type == "atk_def":
        return _extract_poisson_strengths_atk_def(samples_array, samples_columns_mapping,
                                                home_name, away_name, game_indices)
    else:
        return _extract_poisson_strengths_simple(samples_array, samples_columns_mapping,
                                              home_name, away_name, game_indices)

def _simulate_poisson_goals(home_strengths: np.ndarray, away_strengths: np.ndarray,
                           nu_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulate goals using Poisson distribution."""
    home_strengths = np.exp(home_strengths + nu_values)
    away_strengths = np.exp(away_strengths)

    home_goals = np.random.poisson(home_strengths)
    away_goals = np.random.poisson(away_strengths)

    return home_goals, away_goals

def _determine_game_outcomes(home_goals: np.ndarray, away_goals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determine game outcomes from goals."""
    home_win = home_goals > away_goals
    away_win = home_goals < away_goals
    tie = home_goals == away_goals

    return home_win, away_win, tie

def _calculate_poisson_points(home_win: np.ndarray, away_win: np.ndarray, tie: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate points from Poisson game outcomes."""
    home_new_points = home_win * 3 + tie * 1
    away_new_points = away_win * 3 + tie * 1
    return home_new_points, away_new_points

def _calculate_poisson_probabilities(home_win: np.ndarray, away_win: np.ndarray, tie: np.ndarray,
                                   num_simulations: int) -> list[float]:
    """Calculate probabilities for Poisson game outcomes."""
    return [
        float(np.sum(home_win) / num_simulations),
        float(np.sum(tie) / num_simulations),
        float(np.sum(away_win) / num_simulations),
    ]

def generate_points_matrix_poisson(
    samples: pd.DataFrame,
    team_mapping: dict[int, str],
    data: dict[str, Any],
    num_games: int,
    num_simulations: int,
    num_total_matches: int,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """Generate a points matrix for the remainder of the season using the Poisson model."""
    # Extract data
    home_team, away_team, home_team_names, away_team_names = _extract_team_data_for_simulation(data, team_mapping)
    n_teams = len(team_mapping)

    # Initialize matrices
    points_matrix = np.zeros((n_teams, num_total_matches - num_games, num_simulations), dtype=int)
    samples_indices = _generate_random_sample_indices(num_total_matches - num_games, num_simulations, len(samples))
    probabilities: dict[str, dict[str, Any]] = {}

    # Get nu values if available
    samples_columns_mapping = {col: i for i, col in enumerate(samples.columns)}
    nu_idx = samples_columns_mapping.get("nu", None)

    # Simulate each game
    for game_id in range(num_games, num_total_matches):
        game_simulation_idx = game_id - num_games
        game_indices = samples_indices[game_simulation_idx]

        # Get nu values
        if nu_idx is not None:
            nu_values = samples.values[game_indices, nu_idx]
        else:
            nu_values = np.zeros(num_simulations)

        # Extract strengths and simulate
        home_name = home_team_names[game_id]
        away_name = away_team_names[game_id]
        home_strengths, away_strengths = _extract_poisson_strengths(samples, home_name, away_name, game_indices)

        home_goals, away_goals = _simulate_poisson_goals(home_strengths, away_strengths, nu_values)
        home_win, away_win, tie = _determine_game_outcomes(home_goals, away_goals)

        # Calculate points and update matrix
        home_new_points, away_new_points = _calculate_poisson_points(home_win, away_win, tie)
        home_idx = home_team[game_id] - 1
        away_idx = away_team[game_id] - 1

        _update_points_matrix(points_matrix, home_idx, away_idx, game_simulation_idx,
                            home_new_points, away_new_points, team_mapping)

        # Calculate probabilities
        probs = _calculate_poisson_probabilities(home_win, away_win, tie, num_simulations)
        probabilities[str(game_id + 1).zfill(3)] = {
            "home_team": home_name,
            "away_team": away_name,
            "probabilities": probs,
        }

    return points_matrix, probabilities


def _determine_total_matches(n_clubs: int) -> int:
    """Determine total number of matches in championship."""
    return n_clubs * (n_clubs - 1)

def _select_simulation_method(model_name: str) -> str:
    """Select simulation method based on model name."""
    return "bradley_terry" if "bradley_terry" in model_name else "poisson"

def simulate_competition(
    samples: pd.DataFrame,
    team_mapping: dict[int, str],
    model_name: str,
    year: int,
    num_games: int,
    championship: str,
    num_simulations: int = 1_000,
) -> tuple[np.ndarray, dict[str, list[tuple[bool, int]]], dict[str, dict[str, Any]]]:
    """Simulate the remainder of the season using posterior samples from the model."""
    # Load data and determine parameters
    data = load_real_data(year, championship)
    n_clubs = len(team_mapping)
    num_total_matches = _determine_total_matches(n_clubs)

    # Get current scenario
    current_scenario = get_real_points_evolution(data, team_mapping)

    # Select and run simulation method
    simulation_method = _select_simulation_method(model_name)

    if simulation_method == "bradley_terry":
        points_matrix, probabilities = generate_points_matrix_bradley_terry(
            samples, team_mapping, data, num_games, num_simulations, num_total_matches
        )
    else:
        points_matrix, probabilities = generate_points_matrix_poisson(
            samples, team_mapping, data, num_games, num_simulations, num_total_matches
        )

    return points_matrix, current_scenario, probabilities


def _validate_probability_data(data: dict[str, Any], probabilities: dict[str, Any]) -> None:
    """Validate that probability data matches existing data."""
    for game_id, probabilities_data in probabilities.items():
        assert data[game_id]["home_team"] == probabilities_data["home_team"]
        assert data[game_id]["away_team"] == probabilities_data["away_team"]

def _update_game_probabilities(data: dict[str, Any], probabilities: dict[str, Any],
                             model_name: str, num_games: int) -> None:
    """Update game data with new probabilities."""
    for game_id, probabilities_data in probabilities.items():
        data[game_id]["probabilities"] = data[game_id].get("probabilities", {})
        data[game_id]["probabilities"][model_name] = data[game_id]["probabilities"].get(model_name, {})
        data[game_id]["probabilities"][model_name][str(num_games)] = probabilities_data["probabilities"]

def _save_updated_data(data: dict[str, Any], data_path: str) -> None:
    """Save updated data to file."""
    save_json(data, data_path)

def update_probabilities(
    probabilities: dict[str, Any], year: int, model_name: str, num_games: int, championship: str
) -> None:
    """Update the probabilities for a given year, model name, and number of rounds."""
    # Load data
    data, data_path = load_all_matches_data(year, championship)

    # Validate and update
    _validate_probability_data(data, probabilities)
    _update_game_probabilities(data, probabilities, model_name, num_games)

    # Save updated data
    _save_updated_data(data, data_path)


def _calculate_position_probabilities(final_points_distribution: np.ndarray,
                                   team_mapping: dict[int, str]) -> dict[str, list[float]]:
    """Calculate probability of each team finishing in each position."""
    final_positions = np.argsort(final_points_distribution, axis=0)
    n_teams = len(team_mapping)
    final_positions_probs: dict[str, list[float]] = {team: [] for team in team_mapping.values()}

    for idx, team in team_mapping.items():
        for position in range(n_teams):
            prob_team_position = np.mean(final_positions[position, :] == idx - 1)
            final_positions_probs[team].insert(0, prob_team_position)

    return final_positions_probs

def _save_final_positions_probs(final_positions_probs: dict[str, list[float]], save_dir: str) -> None:
    """Save final positions probabilities to JSON file."""
    save_json(final_positions_probs, os.path.join(save_dir, "final_positions_probs.json"))

def calculate_final_positions_probs(
    final_points_distribution: np.ndarray,
    team_mapping: dict[int, str],
    save_dir: str,
) -> None:
    """Calculates the probability of each team finishing in each possible final position."""
    # Calculate probabilities
    final_positions_probs = _calculate_position_probabilities(final_points_distribution, team_mapping)

    # Save results
    _save_final_positions_probs(final_positions_probs, save_dir)
