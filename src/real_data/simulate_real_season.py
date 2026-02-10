# pylint: disable=too-many-arguments, too-many-positional-arguments, duplicate-code, too-many-locals

import json
import os
from datetime import datetime as dt
from itertools import product
from time import time
from typing import Optional

import pandas as pd
from tqdm import tqdm

from data_processing import (
    flush_and_clear_cache,
    generate_all_matches_data,
    generate_real_data_stan_input,
    load_all_matches_data,
)
from model_execution import run_model_with_real_data, set_team_strengths
from simulation import (
    calculate_final_positions_probs,
    calculate_final_positions_real,
    simulate_competition,
    update_probabilities,
)
from visualization import generate_boxplot, generate_points_evolution_by_team
from features.constants import IGNORE_COLS

def fill_matches(base_path: str) -> None:
    """Fill missing matches in the all_matches.json file for a given year.

    This function reads the existing match data, identifies all teams that have played,
    generates all possible match combinations between teams, removes already played matches,
    and adds the remaining matches as "TBD" (To Be Determined) entries to the data.

    Args:
        base_path (str): The base path to the real data.

    Returns:
        None: The function modifies the all_matches.json file in place.
    """
    os.makedirs(base_path, exist_ok=True)
    save_path = os.path.join(base_path, "all_matches.json")
    with open(save_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    teams = set()
    for game_data in data.values():
        teams.add(game_data["home_team"])
        teams.add(game_data["away_team"])

    possible_matches = list(product(teams, teams))
    for game in data.values():
        possible_matches.remove((game["home_team"], game["away_team"]))

    for match in possible_matches:
        if match[0] == match[1]:
            continue
        data[str(len(data) + 1).zfill(3)] = {
            "home_team": match[0],
            "away_team": match[1],
            "goals_team1": None,
            "goals_team2": None,
            "result": "TBD",
        }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def summarize_results(save_dir: str, match_date: str | None) -> None:
    """Summarize the results of the simulation.

    This function reads the final positions probabilities, calculates the probabilities
    of each team finishing in each possible final position, and saves the results in a CSV file.

    Args:
        save_dir (str): The directory where the final positions probabilities are saved.
        match_date (str | None): The date of the last match played.

    Returns:
        None: The function saves the summary results in a CSV file.
    """
    with open(os.path.join(save_dir, "final_positions_probs.json"), "r", encoding="utf-8") as f:
        final_positions_probs = json.load(f)

    df = pd.DataFrame(
        columns=[
            'Date', 'Club', 'Champion',
            'G4', 'G5', 'G6', 'G7', 'G8',
            'Sula (8-13)', 'Sula (9-14)', 'Z4'
        ]
    )
    for team, probs in final_positions_probs.items():
        df.loc[len(df)] = {
            'Date': match_date,
            'Club': team,
            'Champion': probs[0],
            'G4': sum(probs[:4]),
            'G5': sum(probs[:5]),
            'G6': sum(probs[:6]),
            'G7': sum(probs[:7]),
            'G8': sum(probs[:8]),
            'Sula (8-13)': sum(probs[7:13]),
            'Sula (9-14)': sum(probs[8:14]),
            'Z4': sum(probs[-4:])
        }
    df.sort_values(
        by=['Champion', 'G4', 'G5', 'G6', 'G7', 'G8', 'Sula (8-13)', 'Sula (9-14)', 'Z4'],
        ascending=[False, False, False, False, False, False, False, False, True],
        ignore_index=True,
        inplace=True
    )
    df.to_csv(os.path.join(save_dir, "summary_results.csv"), index=False)


def simulate_year(
    year: int,
    model_name: str,
    num_games: int = 380,
    match_date: str | None = None,
    championship: str = "brazil",
    num_simulations: int = 10_000,
    base_path: Optional[str] = None
) -> None:
    """
    Run the specified statistical model (Bradley-Terry or Poisson) using real data
    for a given year and number of rounds, generate a boxplot of team strengths,
    and, if not all rounds are played, simulate the remainder of the season and
    generate a points evolution plot.

    Args:
        model_name (str): The name of the model to run.
        num_games (int, optional): Number of games already played. Defaults to 380.
        match_date (str | None, optional): The date of the last match played. Defaults to None.
        championship (str, optional): The championship to use. Defaults to "brazil".
        num_simulations (int, optional): Number of simulations to run. Defaults to 10000.
        base_path (str, optional): The base path to the real data. Defaults to None.

    Returns:
        None
    """
    generate_real_data_stan_input(year, num_games, championship, base_path)
    fit, team_mapping, model_save_dir = run_model_with_real_data(
        model_name, year, num_games, championship, base_path
    )
    n_clubs = len(team_mapping)
    if n_clubs < 20:
        return

    samples = fit.draws_pd()
    samples = samples.drop(
        columns=[col for col in samples.columns if "raw" in col] + IGNORE_COLS
    )
    samples = set_team_strengths(samples, team_mapping)
    generate_boxplot(
        samples[list(team_mapping.values())],
        year,
        model_save_dir,
        num_games,
    )

    if num_games != n_clubs * (n_clubs - 1):
        points_matrix, current_scenario, probabilities = simulate_competition(
            samples, team_mapping, model_name, year, num_games, championship, num_simulations
        )
        update_probabilities(probabilities, year, model_name, num_games, championship, base_path)
        final_points_distribution = generate_points_evolution_by_team(
            points_matrix,
            current_scenario,
            team_mapping,
            num_games,
            save_dir=model_save_dir,
            make_plots=False,
        )
        calculate_final_positions_probs(
            final_points_distribution,
            team_mapping,
            model_save_dir,
        )

        summarize_results(model_save_dir, match_date)
    else:
        data, _ = load_all_matches_data(year, championship, base_path)
        calculate_final_positions_real(
            data,
            team_mapping,
            model_save_dir,
            num_simulations,
        )
        summarize_results(model_save_dir, match_date)


def run_simulation(year: int, models: list[str], base_path: str, n_simulations: int) -> None:
    """Run the simulation for a given year and models.

    Args:
        year (int): The year of the simulation.
        models (list[str]): The models to run.
        base_path (str): The base path to the real data.
        n_simulations (int): The number of simulations to run.
    """
    os.makedirs(base_path, exist_ok=True)
    generate_all_matches_data(year, "brazil", os.path.join(base_path, "all_matches.json"))
    fill_matches(base_path)

    with open(
        os.path.join(
            base_path, "all_matches.json"
        ),
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    df = pd.DataFrame(data).T.dropna()
    df['match_date'] = df['match_datetime'] \
        .apply(
            lambda x: dt.strptime(str(x).split()[0], "%Y/%m/%d").strftime("%d/%m/%Y")
        )

    df.reset_index(inplace=True)
    df['index'] = df['index'].astype(int)
    games_to_simulate = df \
        .groupby('match_date') \
        .agg({'index': 'max'}) \
        .reset_index() \
        .sort_values(by='index', ignore_index=True)

    games_to_simulate['match_date_dt'] = pd.to_datetime(
        games_to_simulate['match_date'],
        dayfirst=True
    )
    games_to_simulate['next_match_date_dt'] = games_to_simulate['match_date_dt'].shift(-1)
    games_to_simulate['days_to_next_match'] = (
            games_to_simulate['next_match_date_dt'] - games_to_simulate['match_date_dt']
        ).dt.days.fillna(3).astype(int)

    games_to_simulate['consider'] = False
    skip_count = 0
    for i in games_to_simulate.index:
        if games_to_simulate.loc[i]['days_to_next_match'] > 1:
            games_to_simulate.loc[i, 'consider'] = True
            skip_count = 0
        elif skip_count < 2:
            skip_count += 1
        else:
            games_to_simulate.loc[i, 'consider'] = True
            skip_count = 0

    games_to_simulate = games_to_simulate[games_to_simulate['consider']]
    games_to_simulate = dict(zip(games_to_simulate['index'], games_to_simulate['match_date']))
    for model, (game, match_date) in tqdm(product(models, games_to_simulate.items())):
        final_path = os.path.join(
            base_path, model, f"{str(game).zfill(3)}_games", "summary_results.csv"
        )

        if os.path.exists(final_path):
            continue

        simulate_year(
            year=year,
            model_name=model,
            num_games=game,
            match_date=match_date,
            championship="brazil",
            num_simulations=n_simulations,
            base_path=base_path,
        )

    flush_and_clear_cache()


if __name__ == "__main__":
    start_time = time()
    n_simulations = 10_000
    year = 2019
    models = [
        "poisson_2",
        # "poisson_4",
        # "poisson_7",
        # "poisson_9",
    ]
    base_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "real_data",
        "club_level_simulations", "brazil", f"{year}"
    )

    run_simulation(year, models, base_path, n_simulations)

    os.system("clear")
    print(f"Total time elapsed (simulation): {time() - start_time:,.2f} seconds")
