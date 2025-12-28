import numpy as np

from src.features.players_generators import data_generator_poisson


def data_generator_poisson_1(
    *, seed: int | None = None
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model without home advantage."""
    return data_generator_poisson(
        seed=seed,
        home_advantage=False,
        atk_def_strength=False,
        place_params=False,
        bivariate=False,
    )


def data_generator_poisson_2(
    *, seed: int | None = None
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        home_advantage=True,
        atk_def_strength=False,
        place_params=False,
        bivariate=False,
    )


def data_generator_poisson_3(
    *, seed: int | None = None
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        home_advantage=False,
        atk_def_strength=True,
        place_params=False,
        bivariate=False,
    )


def data_generator_poisson_4(
    *, seed: int | None = None
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        home_advantage=True,
        atk_def_strength=True,
        place_params=False,
        bivariate=False,
    )


def data_generator_poisson_6(
    *, seed: int | None = None
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model without home advantage."""
    return data_generator_poisson(
        seed=seed,
        home_advantage=False,
        atk_def_strength=False,
        place_params=False,
        bivariate=True,
    )


def data_generator_poisson_7(
    *, seed: int | None = None
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        home_advantage=True,
        atk_def_strength=False,
        place_params=False,
        bivariate=True,
    )


def data_generator_poisson_8(
    *, seed: int | None = None
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        home_advantage=False,
        atk_def_strength=True,
        place_params=False,
        bivariate=True,
    )


def data_generator_poisson_9(
    *, seed: int | None = None
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Poisson model with home advantage."""
    return data_generator_poisson(
        seed=seed,
        home_advantage=True,
        atk_def_strength=True,
        place_params=False,
        bivariate=True,
    )
