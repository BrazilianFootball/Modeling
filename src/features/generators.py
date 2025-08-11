from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def generate_normal_prior_data(
    *,
    seed: Optional[int] = None,
    n_observations: int = 100,
    true_mu: float = 0,
    true_sigma: float = 1,
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


def generate_matches(clubs: List[int], n_seasons: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create arrays of teams for all matches in all seasons.

    Args:
        clubs: List of club IDs
        n_seasons: Number of seasons to simulate

    Returns:
        Tuple containing arrays of home teams and away teams
    """
    home_teams = np.repeat(clubs, len(clubs))
    away_teams = np.tile(clubs, len(clubs))

    home_teams = np.concatenate([home_teams for _ in range(n_seasons)])
    away_teams = np.concatenate([away_teams for _ in range(n_seasons)])

    mask = home_teams != away_teams
    home_teams = home_teams[mask]
    away_teams = away_teams[mask]

    return home_teams, away_teams


def create_sum_zero_vector(n_elements: int) -> np.ndarray:
    """Create a sum-zero vector of length n_elements."""
    vector = np.random.normal(0, 1, size=n_elements)
    vector[-1] = -np.sum(vector[:-1])
    return vector


def data_generator_bt(  # pylint: disable=too-many-locals
    *,
    seed: Optional[int] = None,
    n_clubs: int = 20,
    n_seasons: int = 1,
    home_advantage: bool = False,
    allow_ties: bool = False,
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Bradley-Terry model with or without home advantage.

    Args:
        seed: Random seed for reproducibility
        n_clubs: Number of clubs to simulate
        n_seasons: Number of seasons to simulate
        home_advantage: Whether to include home advantage
        allow_ties: Whether to allow ties

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    clubs = list(range(1, n_clubs + 1))
    home_teams, away_teams = generate_matches(clubs, n_seasons)

    lambdas = create_sum_zero_vector(n_clubs)
    variables = {"log_skills": lambdas}

    if home_advantage:
        variables["log_home_advantage"] = np.random.normal(0, 1)

    if allow_ties:
        variables["kappa"] = abs(np.random.normal(0, 1))

    home_log_force = lambdas[home_teams - 1]
    away_log_force = lambdas[away_teams - 1]

    home_log_force += variables.get("log_home_advantage", 0)
    kappa = variables.get("kappa", 0)

    home_force = np.exp(home_log_force)
    away_force = np.exp(away_log_force)
    tie_force = kappa * (home_force * away_force) ** (1 / 2)
    total_force = home_force + away_force + tie_force

    prob_home = home_force / total_force
    prob_away = away_force / total_force
    prob_tie = 1 - prob_home - prob_away

    probs = np.column_stack([prob_home, prob_away, prob_tie])
    random_vals = np.random.uniform(size=len(home_teams))
    cumulative_probs = np.cumsum(probs, axis=1)

    results = np.zeros(len(home_teams))
    results[random_vals < cumulative_probs[:, 0]] = 1
    results[
        (random_vals >= cumulative_probs[:, 0]) & (random_vals < cumulative_probs[:, 1])
    ] = 0
    results[random_vals >= cumulative_probs[:, 1]] = 0.5

    data = {
        "home_name": home_teams,
        "home_log_force": home_log_force,
        "away_name": away_teams,
        "away_log_force": away_log_force,
        "log_home_advantage": variables.get("log_home_advantage", 0),
        "kappa": kappa,
        "results": results.tolist() if allow_ties else results.astype(int).tolist(),
    }

    return {
        "variables": variables,
        "generated": {
            "num_games": len(home_teams),
            "num_teams": n_clubs,
            "team1": data["home_name"],
            "team2": data["away_name"],
            "results": data["results"],
        },
    }


def data_generator_bt_1(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Bradley-Terry model without home advantage."""
    return data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        allow_ties=False,
    )


def data_generator_bt_2(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Bradley-Terry model with home advantage."""
    return data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        allow_ties=False,
    )


def data_generator_bt_3(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Bradley-Terry model with ties."""
    return data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        allow_ties=True,
    )


def data_generator_bt_4(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Bradley-Terry model with home advantage and ties."""
    return data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        allow_ties=True,
    )


def mapping_params_to_model(
    home_advantage: bool,
    atk_def_strength: bool,
    place_params: bool,
) -> str:
    """Map parameters to model."""
    if place_params:
        return "poisson_5"

    if atk_def_strength and home_advantage:
        return "poisson_4"

    if atk_def_strength:
        return "poisson_3"

    if home_advantage:
        return "poisson_2"

    return "poisson_1"


def data_generator_poisson(  # pylint: disable=too-many-locals
    *,
    seed: Optional[int] = None,
    n_clubs: int = 20,
    n_seasons: int = 1,
    home_advantage: bool = False,
    atk_def_strength: bool = False,
    place_params: bool = False,
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Poisson model with or without home advantage.

    Args:
        seed: Random seed for reproducibility
        n_clubs: Number of clubs to simulate
        n_seasons: Number of seasons to simulate
        home_advantage: Whether to include home advantage
        atk_def_strength: Whether to include attack and defense strength
        place_params: Indicates whether attack and defense strength depends on place

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    clubs = list(range(1, n_clubs + 1))
    home_teams, away_teams = generate_matches(clubs, n_seasons)

    model_name = mapping_params_to_model(home_advantage, atk_def_strength, place_params)
    alpha = create_sum_zero_vector(n_clubs)
    gamma = create_sum_zero_vector(n_clubs)
    beta = np.random.normal(0, 1, size=n_clubs)
    delta = np.random.normal(0, 1, size=n_clubs)
    nu = np.random.normal(0, 1)
    variables = {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "nu": nu}

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

    if model_name in ["poisson_1", "poisson_2"]:
        home_parameters = np.exp(alpha[home_teams - 1] - alpha[away_teams - 1] + nu)
        away_parameters = np.exp(alpha[away_teams - 1] - alpha[home_teams - 1])
    else:
        home_parameters = np.exp(alpha[home_teams - 1] + beta[away_teams - 1] + nu)
        away_parameters = np.exp(gamma[home_teams - 1] + delta[away_teams - 1])

    home_goals = np.random.poisson(home_parameters)
    away_goals = np.random.poisson(away_parameters)

    return {
        "variables": variables,
        "generated": {
            "num_games": len(home_teams),
            "num_teams": n_clubs,
            "team1": home_teams,
            "team2": away_teams,
            "goals_team1": home_goals,
            "goals_team2": away_goals,
        },
    }


def data_generator_poisson_1(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Poisson model without home advantage."""
    return data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        atk_def_strength=False,
        place_params=False,
    )


def data_generator_poisson_2(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        atk_def_strength=False,
        place_params=False,
    )


def data_generator_poisson_3(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        atk_def_strength=True,
        place_params=False,
    )


def data_generator_poisson_4(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        atk_def_strength=True,
        place_params=False,
    )


def data_generator_poisson_5(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        atk_def_strength=False,
        place_params=True,
    )
