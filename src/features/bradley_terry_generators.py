import numpy as np

from src.features.generators import data_generator_bt


def data_generator_bt_1(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model without home advantage."""
    return data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        allow_ties=False,
    )


def data_generator_bt_2(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with home advantage."""
    return data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        allow_ties=False,
    )


def data_generator_bt_3(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with ties."""
    return data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        allow_ties=True,
    )


def data_generator_bt_4(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with home advantage and ties."""
    return data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        allow_ties=True,
    )
