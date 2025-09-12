# pylint: disable=too-many-locals, too-many-statements

import json
import os
from typing import Any


def generate_all_matches_data(year: int) -> None:
    """
    Generate all matches data for a given year.

    Args:
        year (int): The year of the data to generate.
    """
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "real_data", "results", f"{year}"
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


def generate_real_data_stan_input(year: int, num_rounds: int = 38) -> None:
    """
    Load and process real Serie A game data for a given year and number of rounds,
    and save the processed data as JSON files for use with the Bradley-Terry and Poisson models.

    Args:
        year (int): The year of the games to load and process.
        num_rounds (int, optional): Number of rounds to process. Defaults to 38.
    """
    all_matches_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "real_data", "results", f"{year}", "all_matches.json"
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
        if len(team1) > num_rounds * (num_teams // 2):
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
        "real_data", "inputs", f"{year}"
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


def load_all_matches_data(year: int) -> tuple[dict[str, Any], str]:
    """
    Load the all matches data for a given year.

    Args:
        year (int): The year of the data to load.

    Returns:
        tuple[dict[str, Any], str]: The loaded data dictionary and the path to the data file.
    """
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "real_data", "results", f"{year}", "all_matches.json"
    )
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    return data, data_path


def load_real_data(year: int) -> dict[str, Any]:
    """
    Load the real data for a given year and number of rounds.

    Args:
        year (int): The year of the data to load.

    Returns:
        Dict[str, Any]: The loaded data dictionary.
    """
    try:
        with open(
            os.path.join(
                os.path.dirname(__file__), "..", "..",
                "real_data", "inputs", f"{year}", "poisson_data_38.json"
            ),
            encoding="utf-8",
        ) as f:
            data = json.load(f)
    except FileNotFoundError:
        generate_real_data_stan_input(year, 38)
        with open(
            os.path.join(
                os.path.dirname(__file__), "..", "..",
                "real_data", "inputs", f"{year}", "poisson_data_38.json"
            ),
            encoding="utf-8",
        ) as f:
            data = json.load(f)
    return data


def check_results_exist(model_name: str, year: int, num_rounds: int) -> bool:
    """
    Check if the results for a given model, year, and number of rounds exist.

    Args:
        model_name (str): The name of the model.
        year (int): The year of the data.
        num_rounds (int): The number of rounds of the data.

    Returns:
        bool: True if the results exist, False otherwise.
    """
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "real_data", "results", f"{year}", f"{model_name}", f"round_{str(num_rounds).zfill(2)}"
    )
    return os.path.exists(save_dir)
