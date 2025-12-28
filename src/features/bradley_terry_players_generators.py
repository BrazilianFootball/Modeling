import numpy as np

from src.features.players_generators import data_generator_bt


def data_generator_bt_3(
    *, seed: int | None = None
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with ties."""
    return data_generator_bt(
        seed=seed,
        home_advantage=False,
        allow_ties=True,
    )


def data_generator_bt_4(
    *, seed: int | None = None
) -> dict[str, dict[str, np.ndarray | int]]:
    """Generate data for Bradley-Terry model with home advantage and ties."""
    return data_generator_bt(
        seed=seed,
        home_advantage=True,
        allow_ties=True,
    )
