# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-branches

import json
import os
from typing import Any

import numpy as np

from src.features.generators import mapping_params_to_model


def generate_players_data() -> tuple[dict, dict]:
    """Generate players data.

    Args:
        None

    Returns:
        Tuple containing data and players mapping
    """
    data_path = os.path.join(os.getcwd(), "..", "Data", "results", "processed")

    players_mapping = {"None": 1}
    players_time_on_match: dict[str, int] = {}
    data: dict[str, Any] = {
        "home_players": [],
        "home_players_minutes": [],
        "away_players": [],
        "away_players_minutes": [],
        "home_goals": [],
        "away_goals": [],
        "results": [],
    }

    file_name = "Serie_A_2025_squads.json"
    with open(os.path.join(data_path, file_name), encoding="utf-8") as f:
        real_data = json.load(f)

    for game_data in real_data.values():
        home_goals, away_goals = list(
            map(int, game_data["Summary"]["Result"].upper().split(" X "))
        )
        del game_data["Summary"]
        home_players: dict[str, int] = {}
        away_players: dict[str, int] = {}
        for sub_game_data in game_data.values():
            if sub_game_data["Time"] == 0:
                continue

            sub_game_time = sub_game_data["Time"]
            for player in sub_game_data["Home"]["Squad"]:
                players_mapping[player] = players_mapping.setdefault(
                    player, len(players_mapping) + 1
                )
                players_time_on_match[player] = (
                    players_time_on_match.get(player, 0) + sub_game_time
                )
                home_players[player] = (
                    home_players.get(player, 0) + sub_game_data["Time"]
                )

            for player in sub_game_data["Away"]["Squad"]:
                players_mapping[player] = players_mapping.setdefault(
                    player, len(players_mapping) + 1
                )
                players_time_on_match[player] = (
                    players_time_on_match.get(player, 0) + sub_game_time
                )
                away_players[player] = (
                    away_players.get(player, 0) + sub_game_data["Time"]
                )

        if sum(home_players.values()) < 990:
            home_players["None"] = 990 - sum(home_players.values())
        if sum(away_players.values()) < 990:
            away_players["None"] = 990 - sum(away_players.values())

        data["home_players"].append([players_mapping[x] for x in home_players])
        data["home_players_minutes"].append(list(home_players.values()))
        data["away_players"].append([players_mapping[x] for x in away_players])
        data["away_players_minutes"].append(list(away_players.values()))
        data["home_goals"].append(home_goals)
        data["away_goals"].append(away_goals)
        if home_goals > away_goals:
            data["results"].append(1)
        elif home_goals < away_goals:
            data["results"].append(0)
        else:
            data["results"].append(0.5)
    num_players_per_game = max(
        [len(x) for x in data["home_players"]] + [len(x) for x in data["away_players"]]
    )

    data["num_games"] = len(data["home_goals"])
    data["num_players"] = len(players_mapping)
    data["num_players_per_game"] = num_players_per_game

    for i in range(data["num_games"]):
        while len(data["home_players"][i]) < num_players_per_game:
            data["home_players"][i].append(1)
            data["home_players_minutes"][i].append(0)
        while len(data["away_players"][i]) < num_players_per_game:
            data["away_players"][i].append(1)
            data["away_players_minutes"][i].append(0)

    return data, players_mapping


def create_players_forces(
    players_mapping: dict[str, int],
    variance: float = 1,
) -> np.ndarray:
    """Create a sum-zero vector of length n_elements."""
    players_forces = np.random.normal(0, variance, size=len(players_mapping))
    players_forces[0] = -np.sum(players_forces[1:])
    if players_forces[0] > 0:
        players_forces = -players_forces

    return players_forces


def simulate_bradley_terry(
    home_log_force: np.ndarray,
    away_log_force: np.ndarray,
    kappa: np.ndarray,
) -> np.ndarray:
    """
    Simulate match results using the Bradley-Terry model with ties.

    Args:
        home_log_force (np.ndarray): Log-strengths of the home teams for each match.
        away_log_force (np.ndarray): Log-strengths of the away teams for each match.
        kappa (np.ndarray): Tie parameter controlling the likelihood of draws.

    Returns:
        np.ndarray: Array of simulated match results.
            1   - home win
            0   - away win
            0.5 - tie
    """
    home_force = np.exp(home_log_force)
    away_force = np.exp(away_log_force)
    tie_force = kappa * (home_force * away_force) ** (1 / 2)
    total_force = home_force + away_force + tie_force

    prob_home = home_force / total_force
    prob_away = away_force / total_force
    random_val = np.random.uniform()

    if random_val < prob_home:
        return 1
    if random_val < prob_home + prob_away:
        return 0

    return 0.5


def data_generator_bt(
    *,
    seed: int | None = None,
    home_advantage: bool = False,
    allow_ties: bool = False,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with or without home advantage.

    Args:
        seed: Random seed for reproducibility
        home_advantage: Whether to include home advantage
        allow_ties: Whether to allow ties

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    data, players_mapping = generate_players_data()
    players_forces = create_players_forces(players_mapping)
    variables = {"log_skills": players_forces}

    if home_advantage:
        variables["log_home_advantage"] = np.random.normal(0, 1)

    if allow_ties:
        variables["kappa"] = abs(np.random.normal(0, 1))

    kappa = variables.get("kappa", 0)
    for i in range(data["num_games"]):
        home_log_force = np.dot(
            players_forces[np.array(data["home_players"][i]) - 1],
            np.array(data["home_players_minutes"][i]) / 90,
        )
        away_log_force = np.dot(
            players_forces[np.array(data["away_players"][i]) - 1],
            np.array(data["away_players_minutes"][i]) / 90,
        )
        home_log_force += variables.get("log_home_advantage", 0)
        data["results"][i] = simulate_bradley_terry(
            [home_log_force], [away_log_force], [kappa]
        )

    data.pop("home_goals")
    data.pop("away_goals")
    return {
        "variables": variables,
        "generated": data,
    }


def data_generator_poisson(
    *,
    seed: int | None = None,
    home_advantage: bool = False,
    atk_def_strength: bool = False,
    place_params: bool = False,
    bivariate: bool = False,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with or without home advantage.

    Args:
        seed: Random seed for reproducibility
        home_advantage: Whether to include home advantage
        atk_def_strength: Whether to include attack and defense strength
        place_params: Indicates whether attack and defense strength depends on place

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    variance = 0.25
    data, players_mapping = generate_players_data()

    model_name = mapping_params_to_model(home_advantage, atk_def_strength, place_params)
    alpha = create_players_forces(players_mapping, variance)
    gamma = create_players_forces(players_mapping, variance)
    beta = np.random.normal(0, variance, size=len(players_mapping))
    delta = np.random.normal(0, variance, size=len(players_mapping))
    correlation_strength = np.random.normal(0, 1) if bivariate else 0
    nu = np.random.normal(0, 1)
    variables = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "nu": nu,
        "correlation_strength": correlation_strength,
    }

    if not bivariate:
        variables.pop("correlation_strength")

    if model_name in ["poisson_1", "poisson_3", "poisson_5"]:
        nu = 0
        variables.pop("nu")

    if model_name != "poisson_5":
        delta = alpha
        variables.pop("delta")

    if model_name in ["poisson_1", "poisson_2"]:
        beta = 0
        gamma = 0
        variables.pop("beta")
        variables.pop("gamma")
    elif model_name in ["poisson_3", "poisson_4"]:
        gamma = beta
        variables.pop("gamma")

    for i in range(data["num_games"]):
        if model_name in ["poisson_1", "poisson_2"]:
            home_strength = np.dot(
                alpha[np.array(data["home_players"][i]) - 1],
                np.array(data["home_players_minutes"][i]) / 90,
            )
            away_strength = np.dot(
                alpha[np.array(data["away_players"][i]) - 1],
                np.array(data["away_players_minutes"][i]) / 90,
            )
            home_parameters = np.exp(
                home_strength - away_strength + nu + correlation_strength
            )
            away_parameters = np.exp(
                away_strength - home_strength + correlation_strength
            )
        else:
            home_atk_strength = np.dot(
                alpha[np.array(data["home_players"][i]) - 1],
                np.array(data["home_players_minutes"][i]) / 90,
            )
            away_def_strength = np.dot(
                beta[np.array(data["away_players"][i]) - 1],
                np.array(data["away_players_minutes"][i]) / 90,
            )

            home_def_strength = np.dot(
                gamma[np.array(data["home_players"][i]) - 1],
                np.array(data["home_players_minutes"][i]) / 90,
            )
            away_atk_strength = np.dot(
                delta[np.array(data["away_players"][i]) - 1],
                np.array(data["away_players_minutes"][i]) / 90,
            )
            home_parameters = np.exp(
                home_atk_strength + away_def_strength + nu + correlation_strength
            )
            away_parameters = np.exp(
                home_def_strength + away_atk_strength + correlation_strength
            )

        data["home_goals"][i] = np.random.poisson(home_parameters)
        data["away_goals"][i] = np.random.poisson(away_parameters)

    data.pop("results")
    return {
        "variables": variables,
        "generated": data,
    }
