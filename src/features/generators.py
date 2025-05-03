from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def create_mask(
    clubs: List[int], force: np.ndarray, n_seasons: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create arrays of teams and forces for all matches in all seasons.

    Args:
        clubs: List of club IDs
        force: Array of team forces/skills
        n_seasons: Number of seasons to simulate

    Returns:
        Tuple containing arrays of home teams, away teams, home forces and away forces
    """
    home_teams = np.repeat(clubs, len(clubs))
    away_teams = np.tile(clubs, len(clubs))
    home_force = np.repeat(force, len(clubs))
    away_force = np.tile(force, len(clubs))

    home_teams = np.concatenate([home_teams for _ in range(n_seasons)])
    away_teams = np.concatenate([away_teams for _ in range(n_seasons)])
    home_force = np.concatenate([home_force for _ in range(n_seasons)])
    away_force = np.concatenate([away_force for _ in range(n_seasons)])

    mask = home_teams != away_teams
    home_teams = home_teams[mask]
    away_teams = away_teams[mask]
    home_force = home_force[mask]
    away_force = away_force[mask]

    return home_teams, away_teams, home_force, away_force


def data_generator_bt_1(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Bradley-Terry model without home advantage.

    Args:
        seed: Random seed for reproducibility
        n_clubs: Number of clubs to simulate
        n_seasons: Number of seasons to simulate

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    clubs = list(range(1, n_clubs + 1))
    force = np.random.normal(size=n_clubs)
    force[-1] = -sum(force[:-1])

    mask = create_mask(clubs, force, n_seasons)
    home_teams, away_teams, home_force, away_force = mask

    prob_home = np.exp(home_force) / (np.exp(home_force) + np.exp(away_force))
    home_wins = np.zeros_like(prob_home, dtype=int)

    random_vals = np.random.uniform(size=n_clubs * (n_clubs - 1) * n_seasons)
    home_wins = (random_vals < prob_home).astype(int)

    data = {
        "home_name": home_teams,
        "home_force": home_force,
        "away_name": away_teams,
        "away_force": away_force,
        "home_wins": home_wins,
    }

    return {
        "variables": {"skill": force},
        "generated": {
            "num_games": len(home_teams),
            "num_teams": n_clubs,
            "team1": data["home_name"],
            "team2": data["away_name"],
            "team1_win": data["home_wins"],
        },
    }


def data_generator_bt_2(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Bradley-Terry model with home advantage.

    Args:
        seed: Random seed for reproducibility
        n_clubs: Number of clubs to simulate
        n_seasons: Number of seasons to simulate

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    clubs = list(range(1, n_clubs + 1))
    home_advantage = np.random.normal(0, 1)
    force = np.random.normal(size=n_clubs)
    force[-1] = -sum(force[:-1])

    mask = create_mask(clubs, force, n_seasons)
    home_teams, away_teams, home_force, away_force = mask

    prob_home = np.exp(home_force + home_advantage) / (
        np.exp(home_force + home_advantage) + np.exp(away_force)
    )
    home_wins = np.zeros_like(prob_home, dtype=int)

    random_vals = np.random.uniform(size=n_clubs * (n_clubs - 1) * n_seasons)
    home_wins = (random_vals < prob_home).astype(int)

    data = {
        "home_name": home_teams,
        "home_force": home_force,
        "away_name": away_teams,
        "away_force": away_force,
        "home_wins": home_wins,
    }

    return {
        "variables": {"skill": force, "home_advantage": home_advantage},
        "generated": {
            "num_games": len(home_teams),
            "num_teams": n_clubs,
            "team1": data["home_name"],
            "team2": data["away_name"],
            "team1_win": data["home_wins"],
        },
    }


def data_generator_poisson_1(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Poisson model without home advantage.

    Args:
        seed: Random seed for reproducibility
        n_clubs: Number of clubs to simulate
        n_seasons: Number of seasons to simulate

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    clubs = list(range(1, n_clubs + 1))

    log_forces = np.random.normal(0, 1, size=n_clubs)
    log_forces[-1] = -sum(log_forces[:-1])
    force = np.exp(log_forces)
    mask = create_mask(clubs, force, n_seasons)
    home_teams, away_teams, home_force, away_force = mask

    home_goals = np.random.poisson(home_force / away_force)
    away_goals = np.random.poisson(away_force / home_force)

    return {
        "variables": {"log_skills": log_forces},
        "generated": {
            "num_games": len(home_teams),
            "num_teams": n_clubs,
            "team1": home_teams,
            "team2": away_teams,
            "goals_team1": home_goals,
            "goals_team2": away_goals,
        },
    }


def data_generator_poisson_2(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Poisson model with home advantage.

    Args:
        seed: Random seed for reproducibility
        n_clubs: Number of clubs to simulate
        n_seasons: Number of seasons to simulate

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    clubs = list(range(1, n_clubs + 1))

    log_forces = np.random.normal(0, 1, size=n_clubs)
    log_forces[-1] = -sum(log_forces[:-1])
    force = np.exp(log_forces)
    home_boost = np.random.normal(1, 1)
    while home_boost < 0:
        home_boost = np.random.normal(1, 1)

    mask = create_mask(clubs, force, n_seasons)
    home_teams, away_teams, home_force, away_force = mask

    home_force += home_boost
    home_goals = np.random.poisson(home_force / away_force)
    away_goals = np.random.poisson(away_force / home_force)

    return {
        "variables": {"log_skills": log_forces, "home_force": home_boost},
        "generated": {
            "num_games": len(home_teams),
            "num_teams": n_clubs,
            "team1": home_teams,
            "team2": away_teams,
            "goals_team1": home_goals,
            "goals_team2": away_goals,
        },
    }


def generate_normal_prior_data(
    *,
    seed: Optional[int] = None,
    n_observations: int = 100,
    true_mu: float = 0,
    true_sigma: float = 1
) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
    """Generate data for a model with normal prior.

    Args:
        seed: Random seed for reproducibility
        n_observations: Number of observations to generate
        true_mu: True mean of the normal distribution
        true_sigma: True standard deviation of the normal distribution

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    data = {}
    data["variables"] = {"mu": true_mu}
    data["generated"] = {
        "N": n_observations,
        "y": np.random.normal(true_mu, true_sigma, size=n_observations).tolist(),
    }

    return data
