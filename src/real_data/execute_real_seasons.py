# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-branches

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
            Period(
                start_date=datetime(2019, 6, 3),
                end_date=datetime(2019, 6, 11),
                label="FIFA Date",
                text_x_position=datetime(2019, 6, 7),
            ),
            Period(
                start_date=datetime(2019, 9, 2),
                end_date=datetime(2019, 9, 10),
                label="FIFA Date",
                text_x_position=datetime(2019, 9, 6),
            ),
            Period(
                start_date=datetime(2019, 10, 7),
                end_date=datetime(2019, 10, 15),
                label="FIFA Date",
                text_x_position=datetime(2019, 10, 11),
            ),
            Period(
                start_date=datetime(2019, 11, 11),
                end_date=datetime(2019, 11, 19),
                label="FIFA Date",
                text_x_position=datetime(2019, 11, 15),
            ),
        ],
        title_game_annotations=[],
        relegation_game_annotations=[],
        num_simulations=num_simulations,
        heatmap_num_games=190,
    )


def get_config_2020(base_path: str, num_simulations: int) -> VisualizationConfig:
    """Get the visualization configuration for the 2020 Brazilian Serie A season."""
    return VisualizationConfig(
        year=2020,
        base_path=base_path,
        models={
            'poisson_2': 'Poisson 2',
            'poisson_4': 'Poisson 4',
            'poisson_7': 'Poisson 7',
            'poisson_9': 'Poisson 9',
        },
        title_contenders=[
            ClubStyle('Flamengo / RJ', 'solid'),
            ClubStyle('Internacional / RS', 'dash'),
            ClubStyle('Atlético Mineiro / MG', 'solid'),
            ClubStyle('São Paulo / SP', 'dashdot'),
        ],
        relegation_candidates=[
            ClubStyle('Sport / PE', 'solid'),
            ClubStyle('Fortaleza / CE', 'solid'),
            ClubStyle('Vasco da Gama / RJ', 'solid'),
            ClubStyle('Goiás / GO', 'solid'),
            # ClubStyle('Coritiba / PR', 'dash'),
            # ClubStyle('Botafogo / RJ', 'solid'),
        ],
        periods=[
            Period(
                start_date=datetime(2020, 8, 31),
                end_date=datetime(2020, 9, 8),
                label="FIFA Date",
                text_x_position=datetime(2020, 9, 4),
            ),
            Period(
                start_date=datetime(2020, 10, 5),
                end_date=datetime(2020, 10, 13),
                label="FIFA Date",
                text_x_position=datetime(2020, 10, 9),
            ),
            Period(
                start_date=datetime(2020, 11, 9),
                end_date=datetime(2020, 11, 17),
                label="FIFA Date",
                text_x_position=datetime(2020, 11, 13),
            ),
            Period(
                start_date=datetime(2021, 1, 24),
                end_date=datetime(2021, 2, 1),
                label="FIFA Date",
                text_x_position=datetime(2021, 1, 28),
            ),
        ],
        title_game_annotations=[],
        relegation_game_annotations=[],
        num_simulations=num_simulations,
        heatmap_num_games=192,
    )


def get_config_2021(base_path: str, num_simulations: int) -> VisualizationConfig:
    """Get the visualization configuration for the 2021 Brazilian Serie A season."""
    return VisualizationConfig(
        year=2021,
        base_path=base_path,
        models={
            'poisson_2': 'Poisson 2',
            'poisson_4': 'Poisson 4',
            'poisson_7': 'Poisson 7',
            'poisson_9': 'Poisson 9',
        },
        title_contenders=[
            ClubStyle('Atlético Mineiro / MG', 'solid'),
            ClubStyle('Flamengo / RJ', 'solid'),
            ClubStyle('Palmeiras / SP', 'solid'),
        ],
        relegation_candidates=[
            ClubStyle('Cuiabá / MT', 'solid'),
            ClubStyle('Juventude / RS', 'dash'),
            ClubStyle('Grêmio / RS', 'solid'),
            ClubStyle('Bahia / BA', 'dash'),
            ClubStyle('Sport / PE', 'solid'),
            # ClubStyle('Chapecoense / SC', 'solid'),
        ],
        periods=[
            Period(
                start_date=datetime(2021, 6, 13),
                end_date=datetime(2021, 7, 11),
                label="America's Cup",
                text_x_position=datetime(2021, 6, 26),
            ),
            Period(
                start_date=datetime(2021, 5, 31),
                end_date=datetime(2021, 6, 8),
                label="FIFA Date",
                text_x_position=datetime(2021, 6, 4),
            ),
            Period(
                start_date=datetime(2021, 8, 30),
                end_date=datetime(2021, 9, 7),
                label="FIFA Date",
                text_x_position=datetime(2021, 9, 4),
            ),
            Period(
                start_date=datetime(2021, 10, 4),
                end_date=datetime(2021, 10, 12),
                label="FIFA Date",
                text_x_position=datetime(2021, 10, 8),
            ),
            Period(
                start_date=datetime(2021, 11, 8),
                end_date=datetime(2021, 11, 16),
                label="FIFA Date",
                text_x_position=datetime(2021, 11, 12),
            ),
        ],
        title_game_annotations=[],
        relegation_game_annotations=[],
        num_simulations=num_simulations,
        heatmap_num_games=193,
    )


def get_config_2022(base_path: str, num_simulations: int) -> VisualizationConfig:
    """Get the visualization configuration for the 2022 Brazilian Serie A season."""
    return VisualizationConfig(
        year=2022,
        base_path=base_path,
        models={
            'poisson_2': 'Poisson 2',
            'poisson_4': 'Poisson 4',
            'poisson_7': 'Poisson 7',
            'poisson_9': 'Poisson 9',
        },
        title_contenders=[
            ClubStyle('Palmeiras / SP', 'solid'),
            ClubStyle('Internacional / RS', 'solid'),
            ClubStyle('Fluminense / RJ', 'solid'),
        ],
        relegation_candidates=[
            ClubStyle('Coritiba / PR', 'solid'),
            ClubStyle('Cuiabá / MT', 'dash'),
            ClubStyle('Ceará / CE', 'solid'),
            ClubStyle('Avaí / SC', 'solid'),
            ClubStyle('Atlético Goianiense / GO', 'solid'),
            # ClubStyle('Juventude / RS', 'solid'),
        ],
        periods=[
            # Period(
            #     start_date=datetime(2022, 11, 20),
            #     end_date=datetime(2022, 12, 18),
            #     label="FIFA World Cup",
            #     text_x_position=datetime(2022, 11, 30),
            # ),
            Period(
                start_date=datetime(2022, 5, 30),
                end_date=datetime(2022, 6, 14),
                label="FIFA Date",
                text_x_position=datetime(2022, 6, 8),
            ),
            Period(
                start_date=datetime(2022, 9, 19),
                end_date=datetime(2022, 9, 27),
                label="FIFA Date",
                text_x_position=datetime(2022, 9, 23),
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
            ClubStyle('Goiás / GO', 'solid'),
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


def get_config_2024(base_path: str, num_simulations: int) -> VisualizationConfig:
    """Get the visualization configuration for the 2024 Brazilian Serie A season."""
    return VisualizationConfig(
        year=2024,
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
            ClubStyle('Fortaleza / CE', 'solid'),
        ],
        relegation_candidates=[
            ClubStyle('Grêmio / RS', 'solid'),
            ClubStyle('Juventude / RS', 'solid'),
            ClubStyle('Red Bull Bragantino / SP', 'solid'),
            ClubStyle('Criciúma / SC', 'solid'),
            ClubStyle('Athletico Paranaense / PR', 'dash'),
            # ClubStyle('Atlético Goianiense / GO', 'solid'),
            # ClubStyle('Cuiabá / MT', 'solid'),
        ],
        periods=[
            Period(
                start_date=datetime(2024, 6, 20),
                end_date=datetime(2024, 7, 14),
                label="America's Cup",
                text_x_position=datetime(2024, 7, 2),
            ),
            Period(
                start_date=datetime(2024, 6, 3),
                end_date=datetime(2024, 6, 11),
                label="FIFA Date",
                text_x_position=datetime(2024, 6, 7),
            ),
            Period(
                start_date=datetime(2024, 9, 2),
                end_date=datetime(2024, 9, 10),
                label="FIFA Date",
                text_x_position=datetime(2024, 9, 6),
            ),
            Period(
                start_date=datetime(2024, 10, 7),
                end_date=datetime(2024, 10, 15),
                label="FIFA Date",
                text_x_position=datetime(2024, 10, 11),
            ),
            Period(
                start_date=datetime(2024, 11, 11),
                end_date=datetime(2024, 11, 19),
                label="FIFA Date",
                text_x_position=datetime(2024, 11, 15),
            )
        ],
        title_game_annotations=[],
        relegation_game_annotations=[],
        num_simulations=num_simulations,
        heatmap_num_games=188,
    )


def get_config_2025(base_path: str, num_simulations: int) -> VisualizationConfig:
    """Get the visualization configuration for the 2025 Brazilian Serie A season."""
    return VisualizationConfig(
        year=2025,
        base_path=base_path,
        models={
            'poisson_2': 'Poisson 2',
            'poisson_4': 'Poisson 4',
            'poisson_7': 'Poisson 7',
            'poisson_9': 'Poisson 9',
        },
        title_contenders=[
            ClubStyle('Palmeiras / SP', 'solid'),
            ClubStyle('Flamengo / RJ', 'solid'),
            ClubStyle('Cruzeiro / MG', 'solid'),
        ],
        relegation_candidates=[
            ClubStyle('Vasco da Gama / RJ', 'solid'),
            ClubStyle('Vitória / BA', 'solid'),
            ClubStyle('Internacional / RS', 'dash'),
            ClubStyle('Ceará / CE', 'dash'),
            ClubStyle('Fortaleza / CE', 'solid'),
            # ClubStyle('Juventude / RS', 'solid'),
            # ClubStyle('Sport / PE', 'solid'),
        ],
        periods=[
            Period(
                start_date=datetime(2025, 6, 12),
                end_date=datetime(2025, 7, 12),
                label="FIFA Club World Cup",
                text_x_position=datetime(2025, 6, 27),
            ),
            Period(
                start_date=datetime(2025, 6, 2),
                end_date=datetime(2025, 6, 10),
                label="FIFA Date",
                text_x_position=datetime(2025, 6, 6),
            ),
            Period(
                start_date=datetime(2025, 9, 1),
                end_date=datetime(2025, 9, 9),
                label="FIFA Date",
                text_x_position=datetime(2025, 9, 5),
            ),
            Period(
                start_date=datetime(2025, 10, 6),
                end_date=datetime(2025, 10, 14),
                label="FIFA Date",
                text_x_position=datetime(2025, 10, 10),
            ),
            Period(
                start_date=datetime(2025, 11, 10),
                end_date=datetime(2025, 11, 18),
                label="FIFA Date",
                text_x_position=datetime(2025, 11, 14),
            ),
        ],
        title_game_annotations=[],
        relegation_game_annotations=[],
        num_simulations=num_simulations,
        heatmap_num_games=189,
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

    inner_models = models.copy()
    models_to_remove = []
    for model in models:
        final_result = os.path.join(base_path, model, "380_games", "summary_results.csv")
        if os.path.exists(final_result) and not only_visualization:
            models_to_remove.append(model)

    for model in models_to_remove:
        inner_models.remove(model)

    if len(inner_models) == 0 and not only_visualization:
        return

    if not only_visualization:
        run_simulation(year, inner_models, base_path, num_simulations)

    if year == 2019:
        config = get_config_2019(base_path, num_simulations)
    elif year == 2020:
        config = get_config_2020(base_path, num_simulations)
    elif year == 2021:
        config = get_config_2021(base_path, num_simulations)
    elif year == 2022:
        config = get_config_2022(base_path, num_simulations)
    elif year == 2023:
        config = get_config_2023(base_path, num_simulations)
    elif year == 2024:
        config = get_config_2024(base_path, num_simulations)
    elif year == 2025:
        config = get_config_2025(base_path, num_simulations)
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
    seasons = [*range(2019, 2026)]
    for year in seasons:
        print(f"Running season {year}")
        run_season(year, models, num_simulations, only_visualization=True)

    os.system("clear")
    print(f"Total time elapsed: {time() - start_time:,.2f} seconds")

if __name__ == "__main__":
    main()
