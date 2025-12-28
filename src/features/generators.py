# pylint: disable=too-many-locals, too-many-statements, too-many-arguments

import numpy as np


def generate_matches(clubs: list[int], n_seasons: int) -> tuple[np.ndarray, np.ndarray]:
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


def create_sum_zero_vector(n_elements: int, variance: float = 1) -> np.ndarray:
    """Create a sum-zero vector of length n_elements."""
    vector = np.random.normal(0, variance, size=n_elements)
    vector[-1] = -np.sum(vector[:-1])
    return vector


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
    prob_tie = 1 - prob_home - prob_away

    probs = np.column_stack([prob_home, prob_away, prob_tie])
    random_vals = np.random.uniform(size=len(home_log_force))
    cumulative_probs = np.cumsum(probs, axis=1)

    results = np.zeros(len(home_log_force))
    results[random_vals < cumulative_probs[:, 0]] = 1
    results[
        (random_vals >= cumulative_probs[:, 0]) & (random_vals < cumulative_probs[:, 1])
    ] = 0
    results[random_vals >= cumulative_probs[:, 1]] = 0.5

    return results


def data_generator_bt(
    *,
    seed: int | None = None,
    n_clubs: int = 20,
    n_seasons: int = 1,
    home_advantage: bool = False,
    allow_ties: bool = False,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with or without home advantage.

    Args:
        seed: Random seed for reproducibility
        n_clubs: Number of clubs to simulate
        n_seasons: Number of seasons to simulate
        home_advantage: Whether to include home advantage
        allow_ties: Whether to allow ties
        n_players_per_club: Number of players per club

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    clubs = list(range(1, n_clubs + 1))
    home_teams, away_teams = generate_matches(clubs, n_seasons)

    lambdas = create_sum_zero_vector(n_clubs)
    variables = {"log_skills": lambdas}

    home_log_force = lambdas[home_teams - 1]
    away_log_force = lambdas[away_teams - 1]

    if home_advantage:
        variables["log_home_advantage"] = np.random.normal(0, 1)

    if allow_ties:
        variables["kappa"] = abs(np.random.normal(0, 1))

    home_log_force += variables.get("log_home_advantage", 0)
    kappa = variables.get("kappa", 0) * np.ones(len(home_teams))
    results = simulate_bradley_terry(home_log_force, away_log_force, kappa)

    return {
        "variables": variables,
        "generated": {
            "num_games": len(home_teams),
            "num_teams": n_clubs,
            "team1": home_teams,
            "team2": away_teams,
            "results": results.tolist() if allow_ties else results.astype(int).tolist(),
        },
    }


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


def data_generator_poisson(
    *,
    seed: int | None = None,
    n_clubs: int = 20,
    n_seasons: int = 1,
    home_advantage: bool = False,
    atk_def_strength: bool = False,
    place_params: bool = False,
    bivariate: bool = False,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with or without home advantage.

    Args:
        seed: Random seed for reproducibility
        n_clubs: Number of clubs to simulate
        n_seasons: Number of seasons to simulate
        home_advantage: Whether to include home advantage
        atk_def_strength: Whether to include attack and defense strength
        place_params: Indicates whether attack and defense strength depends on place
        bivariate: Whether to include bivariate Poisson model

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    clubs = list(range(1, n_clubs + 1))
    home_teams, away_teams = generate_matches(clubs, n_seasons)
    model_name = mapping_params_to_model(home_advantage, atk_def_strength, place_params)
    alpha = create_sum_zero_vector(n_clubs, 1)
    gamma = create_sum_zero_vector(n_clubs, 1)
    beta = np.random.normal(0, 1, size=n_clubs)
    delta = np.random.normal(0, 1, size=n_clubs)
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

    if model_name in ["poisson_1", "poisson_2"]:
        home_strength = np.sum(alpha[home_teams - 1].reshape(1, -1), axis=0)
        away_strength = np.sum(alpha[away_teams - 1].reshape(1, -1), axis=0)
        home_parameters = np.exp(
            home_strength - away_strength + nu + correlation_strength
        )
        away_parameters = np.exp(away_strength - home_strength + correlation_strength)
    else:
        home_atk_strength = np.sum(alpha[home_teams - 1].reshape(1, -1), axis=0)
        away_def_strength = np.sum(beta[away_teams - 1].reshape(1, -1), axis=0)

        home_def_strength = np.sum(gamma[home_teams - 1].reshape(1, -1), axis=0)
        away_atk_strength = np.sum(delta[away_teams - 1].reshape(1, -1), axis=0)
        home_parameters = np.exp(
            home_atk_strength + away_def_strength + nu + correlation_strength
        )
        away_parameters = np.exp(
            home_def_strength + away_atk_strength + correlation_strength
        )

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
