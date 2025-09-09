import numpy as np

from src.features.generators import data_generator_bt


def data_generator_bt_1(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model without home advantage."""
    data = data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        allow_ties=False,
    )

    data["generated"].pop("num_players_per_club")
    data["generated"].pop("team1_players")
    data["generated"].pop("team2_players")

    return data


def data_generator_bt_2(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with home advantage."""
    data = data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        allow_ties=False,
    )

    data["generated"].pop("num_players_per_club")
    data["generated"].pop("team1_players")
    data["generated"].pop("team2_players")

    return data


def data_generator_bt_3(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with ties."""
    data = data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        allow_ties=True,
    )

    data["generated"].pop("num_players_per_club")
    data["generated"].pop("team1_players")
    data["generated"].pop("team2_players")

    return data


def data_generator_bt_4(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with home advantage and ties."""
    data = data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        allow_ties=True,
    )

    data["generated"].pop("num_players_per_club")
    data["generated"].pop("team1_players")
    data["generated"].pop("team2_players")

    return data


def data_generator_bt_5(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model without home advantage."""
    data = data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        allow_ties=False,
        n_players_per_club=27,
    )

    return data


def data_generator_bt_6(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with home advantage."""
    data = data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        allow_ties=False,
        n_players_per_club=27,
    )

    return data


def data_generator_bt_7(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with ties."""
    data = data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        allow_ties=True,
        n_players_per_club=27,
    )

    return data


def data_generator_bt_8(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with home advantage and ties."""
    data = data_generator_bt(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        allow_ties=True,
        n_players_per_club=27,
    )

    return data
