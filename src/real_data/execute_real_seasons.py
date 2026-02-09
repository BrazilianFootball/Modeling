# pylint: disable=too-many-arguments, too-many-positional-arguments

import logging
import os
from time import time
from datetime import datetime

from simulate_real_season import run_simulation
from visualization_real_season import (
    ClubStyle,
    Period,
    VisualizationConfig,
    run_visualization,
)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def get_config_2019(base_path: str, num_simulations: int) -> VisualizationConfig:
    """Get the visualization configuration for the 2019 Brazilian Serie A season.

    Args:
        base_path (str): The base path to the real data.
        num_simulations (int): The number of simulations to run.
    """
    return VisualizationConfig(
        year=2019,
        base_path=base_path,
        models={
            'poisson_2': 'Poisson 2',
            'poisson_4': 'Poisson 4',
            'poisson_7': 'Poisson 7',
            'poisson_9': 'Poisson 9',
        },
        title_contenders=[
            ClubStyle('Flamengo / RJ', 'solid'),
            ClubStyle('Palmeiras / SP', 'solid'),
            ClubStyle('Santos / SP', 'solid'),
        ],
        relegation_candidates=[
            ClubStyle('Botafogo / RJ', 'solid'),
            ClubStyle('Ceará / CE', 'dash'),
            ClubStyle('Cruzeiro / MG', 'solid'),
            ClubStyle('CSA / AL', 'dash'),
            # ClubStyle('Chapecoense / SC', 'dashdot'),
            # ClubStyle('Avaí / SC', 'dashdot'),
        ],
        periods=[
            Period(
                start_date=datetime(2019, 6, 15),
                end_date=datetime(2019, 7, 8),
                label="America's Cup",
                text_x_position=datetime(2019, 6, 26),
            ),

        ],
        title_game_annotations=[],
        relegation_game_annotations=[],
        num_simulations=num_simulations,
        heatmap_num_games=190,
    )


def get_config_2023(base_path: str, num_simulations: int) -> VisualizationConfig:
    """Get the visualization configuration for the 2023 Brazilian Serie A season.

    Args:
        base_path (str): The base path to the real data.
        num_simulations (int): The number of simulations to run.
    """
    return VisualizationConfig(
        year=2023,
        base_path=base_path,
        models={
            'poisson_2': 'Poisson 2',
            'poisson_4': 'Poisson 4',
            'poisson_7': 'Poisson 7',
            'poisson_9': 'Poisson 9',
        },
        title_contenders=[
            ClubStyle('Botafogo / RJ', 'solid'),
            ClubStyle('Palmeiras / SP', 'solid'),
            ClubStyle('Flamengo / RJ', 'solid'),
        ],
        relegation_candidates=[
            ClubStyle('Vasco da Gama / RJ', 'solid'),
            ClubStyle('Bahia / BA', 'solid'),
            ClubStyle('Santos / SP', 'dot'),
            ClubStyle('Goiás / GO', 'dashdot'),
            # ClubStyle('Coritiba / PR', 'dashdot'),
            # ClubStyle('América / MG', 'dashdot'),
        ],
        periods=[
            Period(
                start_date=datetime(2023, 6, 12),
                end_date=datetime(2023, 6, 20),
                label="FIFA Date",
                text_x_position=datetime(2023, 6, 16),
            ),
            Period(
                start_date=datetime(2023, 9, 4),
                end_date=datetime(2023, 9, 12),
                label="FIFA Date",
                text_x_position=datetime(2023, 9, 8),
            ),
            Period(
                start_date=datetime(2023, 10, 9),
                end_date=datetime(2023, 10, 17),
                label="FIFA Date",
                text_x_position=datetime(2023, 10, 13),
            ),
            Period(
                start_date=datetime(2023, 11, 13),
                end_date=datetime(2023, 11, 21),
                label="FIFA Date",
                text_x_position=datetime(2023, 11, 17),
            ),
        ],
        title_game_annotations=[],
        relegation_game_annotations=[],
        num_simulations=num_simulations,
        heatmap_num_games=188,
    )


def run_season(
    year: int,
    models: list[str],
    num_simulations: int,
    only_visualization: bool = False
) -> None:
    """
    Run the complete pipeline (simulation + visualization) for a given season.

    Args:
        year: The year of the season.
        models: List of model names to use.
        num_simulations: Number of simulations to run.
        only_visualization: Whether to only run visualization.
    """
    base_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "real_data", "club_level_simulations", "brazil", f"{year}"
    )

    if not only_visualization:
        run_simulation(year, models, base_path, num_simulations)

    if year == 2019:
        config = get_config_2019(base_path, num_simulations)
    elif year == 2023:
        config = get_config_2023(base_path, num_simulations)
    else:
        raise ValueError(f"No configuration defined for year {year}")

    run_visualization(config)


def main():
    """Main function to execute simulations for all configured seasons."""
    start_time = time()

    models = [
        "poisson_2",
        "poisson_4",
        "poisson_7",
        "poisson_9",
    ]
    num_simulations = 10_000
    seasons = [2019, 2023]
    for year in seasons:
        run_season(year, models, num_simulations, only_visualization=True)

    os.system("clear")
    print(f"Total time elapsed: {time() - start_time:,.2f} seconds")

if __name__ == "__main__":
    main()
