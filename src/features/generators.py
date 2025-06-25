from itertools import product
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


def data_generator_bt(  # pylint: disable=too-many-locals
    *,
    seed: Optional[int] = None,
    n_clubs: int = 20,
    n_seasons: int = 1,
    home_advantage: bool = False
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Bradley-Terry model with or without home advantage.

    Args:
        seed: Random seed for reproducibility
        n_clubs: Number of clubs to simulate
        n_seasons: Number of seasons to simulate
        home_advantage: Whether to include home advantage

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    clubs = list(range(1, n_clubs + 1))
    force = np.random.normal(size=n_clubs)
    force -= np.mean(force)

    variables = {"skills": force}

    if home_advantage:
        home_adv = np.random.normal(0, 1)
        variables["home_advantage"] = home_adv
    else:
        home_adv = 0

    mask = create_mask(clubs, force, n_seasons)
    home_teams, away_teams, home_force, away_force = mask

    prob_home = np.exp(home_force + home_adv) / (
        np.exp(home_force + home_adv) + np.exp(away_force)
    )

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
        "variables": variables,
        "generated": {
            "num_games": len(home_teams),
            "num_teams": n_clubs,
            "team1": data["home_name"],
            "team2": data["away_name"],
            "team1_win": data["home_wins"],
        },
    }


def data_generator_bt_1(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Bradley-Terry model without home advantage."""
    return data_generator_bt(
        seed=seed, n_clubs=n_clubs, n_seasons=n_seasons, home_advantage=False
    )


def data_generator_bt_2(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Bradley-Terry model with home advantage."""
    return data_generator_bt(
        seed=seed, n_clubs=n_clubs, n_seasons=n_seasons, home_advantage=True
    )


def data_generator_poisson(  # pylint: disable=too-many-locals
    *,
    seed: Optional[int] = None,
    n_clubs: int = 20,
    n_seasons: int = 1,
    home_advantage: bool = False
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Poisson model with or without home advantage.

    Args:
        seed: Random seed for reproducibility
        n_clubs: Number of clubs to simulate
        n_seasons: Number of seasons to simulate
        home_advantage: Whether to include home advantage

    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    clubs = list(range(1, n_clubs + 1))

    log_forces = np.random.normal(0, 1, size=n_clubs)
    log_forces -= np.mean(log_forces)
    force = np.exp(log_forces)

    variables = {"log_skills": log_forces}

    if home_advantage:
        log_home_adv = np.random.normal(0, 1)
        variables["log_home_advantage"] = log_home_adv
    else:
        log_home_adv = 0

    mask = create_mask(clubs, force, n_seasons)
    home_teams, away_teams, home_force, away_force = mask

    home_force *= np.exp(log_home_adv)
    home_goals = np.random.poisson(home_force / away_force)
    away_goals = np.random.poisson(away_force / home_force)

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
        seed=seed, n_clubs=n_clubs, n_seasons=n_seasons, home_advantage=False
    )


def data_generator_poisson_2(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed, n_clubs=n_clubs, n_seasons=n_seasons, home_advantage=True
    )


def generate_mask_kn_model(  # pylint: disable=too-many-arguments, too-many-locals
    n_seasons: int,
    mu: float,
    h: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    ha1: np.ndarray,
    hd1: np.ndarray,
    ha2: np.ndarray,
    hd2: np.ndarray,
) -> np.ndarray:
    """Generate parameter array for Karlis and Ntzoufras model.

    Args:
        mu: Mean of the normal distribution
        h: Home/away effect
        a: Offensive performance
        d: Defensive performance
        ha1: Offensive performance at home
        hd1: Defensive performance at home
        ha2: Offensive performance away
        hd2: Defensive performance away

    Returns:
        Parameter array for Karlis and Ntzoufras model
    """

    n_clubs = len(a)
    mask = np.zeros((n_clubs * (n_clubs - 1) * n_seasons, 4))
    for s, i, j in product(range(n_seasons), range(n_clubs), range(n_clubs)):
        if i != j:
            lambda_1 = np.exp(mu + h[0] + a[i] + d[j] + ha1[i] + hd2[j])
            lambda_2 = np.exp(mu + h[1] + a[j] + d[i] + ha2[j] + hd1[i])
            mask[s * n_clubs * (n_clubs - 1) + i * (n_clubs - 1) + j] = np.array(
                [i, j, lambda_1, lambda_2]
            )

    return mask


def data_generator_karlis_and_ntzoufras(  # pylint: disable=too-many-locals
    *,
    seed: Optional[int] = None,
    n_clubs: int = 20,
    n_seasons: int = 1,
    v2: bool = False
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Karlis and Ntzoufras model.

    Args:
        seed: Random seed for reproducibility
        n_clubs: Number of clubs to simulate
        n_seasons: Number of seasons to simulate
        v2: Whether to use the second version of the model
    Returns:
        Dictionary containing variables and generated data
    """
    if seed is not None:
        np.random.seed(seed)

    mu = np.random.normal(0, 1)
    h = np.random.normal(0, 1, size=2)
    a = np.random.normal(0, 1, size=n_clubs)
    d = np.random.normal(0, 1, size=n_clubs)
    h[0] = 0
    a[-1] = -sum(a[:-1])
    d[-1] = -sum(d[:-1])

    ha1 = np.random.normal(0, 1, size=n_clubs)
    hd1 = np.random.normal(0, 1, size=n_clubs)
    ha2 = np.random.normal(0, 1, size=n_clubs)
    hd2 = np.random.normal(0, 1, size=n_clubs)
    if v2:
        ha1 = np.zeros_like(ha1)
        hd1 = np.zeros_like(hd1)
        ha2 = np.zeros_like(ha2)
        hd2 = np.zeros_like(hd2)
    else:
        ha1[0] = 0
        hd1[0] = 0
        ha2[-1] = -sum(ha2[:-1])
        hd2[-1] = -sum(hd2[:-1])

    mask = generate_mask_kn_model(n_seasons, mu, h, a, d, ha1, hd1, ha2, hd2)
    home_teams, away_teams, home_force, away_force = (
        mask[:, 0],
        mask[:, 1],
        mask[:, 2],
        mask[:, 3],
    )

    home_goals = np.random.poisson(home_force)
    away_goals = np.random.poisson(away_force)

    return {
        "variables": {
            "mu": mu,
            "h": h,
            "a": a,
            "d": d,
            "ha1": ha1,
            "hd1": hd1,
            "ha2": ha2,
            "hd2": hd2,
        },
        "generated": {
            "num_games": len(home_teams),
            "num_teams": n_clubs,
            "team1": home_teams.astype(int) + 1,
            "team2": away_teams.astype(int) + 1,
            "goals_team1": home_goals.astype(int),
            "goals_team2": away_goals.astype(int),
        },
    }


def data_generator_karlis_and_ntzoufras_v2(
    *, seed: Optional[int] = None, n_clubs: int = 20, n_seasons: int = 1
) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    """Generate data for Karlis and Ntzoufras model v2."""
    return data_generator_karlis_and_ntzoufras(
        seed=seed, n_clubs=n_clubs, n_seasons=n_seasons, v2=True
    )


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
