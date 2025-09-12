# pylint: disable=too-many-locals, too-many-statements

import json
import os
from typing import Any
import pandas as pd


def generate_all_matches_from_scraped_data(year: int) -> None:
    """
    Generate all matches data for a given year.

    Args:
        year (int): The year of the data to generate.
    """
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "real_data", "results", "brazil", f"{year}"
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "all_matches.json")
    if os.path.exists(save_path):
        return

    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "Data", "results", "processed"
    )
    file_path = os.path.join(path, f"Serie_A_{year}_games.json")

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    all_matches = {}
    for game_id, game_data in data.items():
        home_team = game_data.get("Home")
        away_team = game_data.get("Away")
        result = game_data.get("Result")
        goals_team1, goals_team2 = map(int, result.lower().split(" x "))

        all_matches[game_id] = {
            "home_team": home_team,
            "away_team": away_team,
            "goals_team1": goals_team1,
            "goals_team2": goals_team2,
        }
        if goals_team1 > goals_team2:
            all_matches[game_id]["result"] = "H"
        elif goals_team1 < goals_team2:
            all_matches[game_id]["result"] = "A"
        else:
            all_matches[game_id]["result"] = "D"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_matches, f, ensure_ascii=False, indent=2)


def generate_all_matches_from_football_data_co_uk(year: int, championship: str) -> None:
    """
    Download, process, and save all matches data for a given year and championship
    from football-data.co.uk.

    This function fetches the CSV file for the specified championship and season,
    processes the relevant columns, and saves the data as a JSON file in the
    appropriate directory structure.

    Args:
        year (int): The year of the season to process (e.g., 2022 for the 2022/2023 season).
        championship (str): The championship code (e.g., "england").

    Raises:
        KeyError: If the championship is not supported.
        Exception: If there is an error downloading or processing the data.
    """
    year_mask = str(year % 100) + str(year % 100 + 1)
    championship_mask = {
        "england": "E0",
        "germany": "D1",
        "italy": "I1",
        "spain": "SP1",
        "france": "F1",
    }[championship]
    url = f"https://www.football-data.co.uk/mmz4281/{year_mask}/{championship_mask}.csv"
    df = pd.read_csv(url)
    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
    df.reset_index(inplace=True)
    df.rename(
        columns={
            'index': 'game_id',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FTHG': 'goals_team1',
            'FTAG': 'goals_team2',
            'FTR': 'result'
        },
        inplace=True
    )
    df['game_id'] = (df['game_id'] + 1).astype(str).str.zfill(3)
    data = df.to_dict(orient='records')
    data = {game['game_id']: {k: v for k, v in game.items() if k != 'game_id'} for game in data}

    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "real_data", "results", f"{championship}", f"{year}"
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "all_matches.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_all_matches_data(year: int, championship: str) -> None:
    """
    Generate and save all matches data for a given year and championship.

    This function creates a JSON file containing all match data for the specified
    year and championship. For the Brazilian championship, it uses scraped data.
    For other supported championships, it downloads and processes data from
    football-data.co.uk. If the championship is not supported, it raises a ValueError.

    Args:
        year (int): The year of the matches to process.
        championship (str): The name of the championship (e.g., "brazil", "england").

    Raises:
        ValueError: If the championship is not supported or if there is an error
            processing the data.
    """
    if championship == "brazil":
        generate_all_matches_from_scraped_data(year)
    else:
        try:
            generate_all_matches_from_football_data_co_uk(year, championship)
        except Exception as e:
            raise ValueError(f"Championship {championship} is not supported") from e


def generate_real_data_stan_input(
    year: int,
    num_rounds: int = 38,
    championship: str = "brazil"
) -> None:
    """
    Load and process real Serie A game data for a given year and number of rounds,
    and save the processed data as JSON files for use with the Bradley-Terry and Poisson models.

    Args:
        year (int): The year of the games to load and process.
        num_rounds (int, optional): Number of rounds to process. Defaults to 38.
        championship (str, optional): The championship of the data. Defaults to "brazil".
    """
    all_matches_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "real_data", "results", f"{championship}", f"{year}", "all_matches.json"
    )

    with open(all_matches_path, encoding="utf-8") as f:
        data = json.load(f)

    team_to_index: dict[str, int] = {}
    team_names: list[str] = []

    team1: list[int] = []
    team2: list[int] = []
    goals_team1_list: list[int] = []
    goals_team2_list: list[int] = []
    results: list[float] = []

    # hardcoded start because we don't know the number of teams in advance
    if len(data) == 380:
        num_teams = 20
    elif len(data) == 306:
        num_teams = 18
    else:
        raise ValueError(f"Number of total matches is {len(data)}, which is not supported")

    for game_data in data.values():
        if len(team1) >= num_rounds * (num_teams // 2):
            break

        home_team = game_data.get("home_team")
        away_team = game_data.get("away_team")
        goals_team1 = game_data.get("goals_team1")
        goals_team2 = game_data.get("goals_team2")
        result = game_data.get("result")

        if home_team not in team_to_index:
            team_names.append(home_team)
            team_to_index[home_team] = len(team_names)

        if away_team not in team_to_index:
            team_names.append(away_team)
            team_to_index[away_team] = len(team_names)

        team1.append(team_to_index[home_team])
        team2.append(team_to_index[away_team])
        goals_team1_list.append(goals_team1)
        goals_team2_list.append(goals_team2)

        if result == "H":
            results.append(1.0)
        elif result == "A":
            results.append(0.0)
        else:
            results.append(0.5)

    base_data = {
        "num_games": len(team1),
        "num_teams": num_teams,
        "team1": team1,
        "team2": team2,
        "team_names": team_names,
    }

    bradley_terry_data = base_data.copy()
    bradley_terry_data.update({"results": results})

    poisson_data = base_data.copy()
    poisson_data.update(
        {
            "goals_team1": goals_team1_list,
            "goals_team2": goals_team2_list,
        }
    )

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "real_data", "inputs", f"{championship}", f"{year}"
    )
    os.makedirs(output_dir, exist_ok=True)

    bradley_terry_path = os.path.join(
        output_dir, f"bradley_terry_data_{str(num_rounds).zfill(2)}.json"
    )
    poisson_path = os.path.join(output_dir, f"poisson_data_{str(num_rounds).zfill(2)}.json")

    with open(bradley_terry_path, "w", encoding="utf-8") as f:
        json.dump(bradley_terry_data, f, ensure_ascii=False, indent=2)
    with open(poisson_path, "w", encoding="utf-8") as f:
        json.dump(poisson_data, f, ensure_ascii=False, indent=2)


def load_all_matches_data(year: int, championship: str) -> tuple[dict[str, Any], str]:
    """
    Load the all matches data for a given year.

    Args:
        year (int): The year of the data to load.
        championship (str): The championship of the data.

    Returns:
        tuple[dict[str, Any], str]: The loaded data dictionary and the path to the data file.
    """
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "real_data", "results", f"{championship}", f"{year}", "all_matches.json"
    )
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    return data, data_path


def load_real_data(year: int, championship: str) -> dict[str, Any]:
    """
    Load the real data for a given year and number of rounds.

    Args:
        year (int): The year of the data to load.
        championship (str): The championship of the data.

    Returns:
        Dict[str, Any]: The loaded data dictionary.
    """
    try:
        with open(
            os.path.join(
                os.path.dirname(__file__), "..", "..",
                "real_data", "inputs", f"{championship}", f"{year}", "poisson_data_38.json"
            ),
            encoding="utf-8",
        ) as f:
            data = json.load(f)
    except FileNotFoundError:
        generate_real_data_stan_input(year, 38, championship)
        with open(
            os.path.join(
                os.path.dirname(__file__), "..", "..",
                "real_data", "inputs", f"{championship}", f"{year}", "poisson_data_38.json"
            ),
            encoding="utf-8",
        ) as f:
            data = json.load(f)
    return data


def check_results_exist(model_name: str, year: int, num_rounds: int, championship: str) -> bool:
    """
    Check if the results for a given model, year, and number of rounds exist.

    Args:
        model_name (str): The name of the model.
        year (int): The year of the data.
        num_rounds (int): The number of rounds of the data.
        championship (str): The championship of the data.

    Returns:
        bool: True if the results exist, False otherwise.
    """
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "real_data", "results", f"{championship}", f"{year}",
        f"{model_name}", f"round_{str(num_rounds).zfill(2)}"
    )
    return os.path.exists(save_dir)
