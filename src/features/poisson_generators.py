import numpy as np

from src.features.generators import data_generator_poisson


def data_generator_poisson_1(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model without home advantage."""
    data = data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        atk_def_strength=False,
        place_params=False,
    )

    data["generated"].pop("num_players_per_club")
    data["generated"].pop("team1_players")
    data["generated"].pop("team2_players")

    return data


def data_generator_poisson_2(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    data = data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        atk_def_strength=False,
        place_params=False,
    )

    data["generated"].pop("num_players_per_club")
    data["generated"].pop("team1_players")
    data["generated"].pop("team2_players")

    return data


def data_generator_poisson_3(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    data = data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        atk_def_strength=True,
        place_params=False,
    )

    data["generated"].pop("num_players_per_club")
    data["generated"].pop("team1_players")
    data["generated"].pop("team2_players")

    return data


def data_generator_poisson_4(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    data = data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        atk_def_strength=True,
        place_params=False,
    )

    data["generated"].pop("num_players_per_club")
    data["generated"].pop("team1_players")
    data["generated"].pop("team2_players")

    return data


def data_generator_poisson_5(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    data = data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        atk_def_strength=False,
        place_params=True,
    )

    data["generated"].pop("num_players_per_club")
    data["generated"].pop("team1_players")
    data["generated"].pop("team2_players")

    return data


def data_generator_poisson_6(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model without home advantage."""
    return data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        atk_def_strength=False,
        place_params=False,
        n_players_per_club=27,
    )


def data_generator_poisson_7(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        atk_def_strength=False,
        place_params=False,
        n_players_per_club=27,
    )


def data_generator_poisson_8(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        atk_def_strength=True,
        place_params=False,
        n_players_per_club=27,
    )


def data_generator_poisson_9(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=True,
        atk_def_strength=True,
        place_params=False,
        n_players_per_club=27,
    )


def data_generator_poisson_10(
    *, seed: int | None = None, n_clubs: int = 20, n_seasons: int = 1
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        n_clubs=n_clubs,
        n_seasons=n_seasons,
        home_advantage=False,
        atk_def_strength=False,
        place_params=True,
        n_players_per_club=27,
    )
