# pylint: disable=too-many-locals, too-many-statements

import json
import os
from typing import Any
from datetime import datetime as dt
import pandas as pd

from ..utils.io_utils import save_json, load_json
from ..utils.path_utils import get_results_path, get_inputs_path

def parse_datetime(date: str, time: str) -> str:
    """
    Converts a date and time string to the format "YYYY/MM/DD HH:MM".

    Args:
        date (str): Date in the format "DD/MM/YYYY".
        time (str): Time in the format "HH:MM".

    Returns:
        str: Date and time formatted as "YYYY/MM/DD HH:MM".
             If conversion fails, returns the original concatenated date and time string.
    """
    return dt.strptime(f"{date} {time}", "%d/%m/%Y %H:%M").strftime("%Y/%m/%d %H:%M")


def _load_scraped_data(year: int) -> dict[str, Any]:
    """Load scraped data from file."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..",
        "Data", "results", "processed"
    )
    file_path = os.path.join(path, f"Serie_A_{year}_games.json")

    return load_json(file_path)

def _process_scraped_data(raw_data: dict[str, Any]) -> dict[str, Any]:
    """Process scraped data into standardized format."""
    # Add datetime parsing
    data = {
        game_id: {**game_data, "Datetime": parse_datetime(game_data["Date"], game_data["Time"])}
        for game_id, game_data in raw_data.items()
    }

    # Sort by datetime
    data = sorted(data.items(), key=lambda x: x[1]["Datetime"])

    # Convert to standardized format
    all_matches = {}
    for i, (_, game_data) in enumerate(data):
        game_id = str(i+1).zfill(3)
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

    return all_matches

def _save_all_matches_data(data: dict[str, Any], year: int, championship: str) -> None:
    """Save all matches data to JSON file."""
    save_dir = get_results_path(championship, year)
    save_path = os.path.join(save_dir, "all_matches.json")
    save_json(data, save_path)

def generate_all_matches_from_scraped_data(year: int) -> None:
    """Generate all matches data for a given year."""
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "real_data", "results", "brazil", f"{year}"
    )
    save_path = os.path.join(save_dir, "all_matches.json")

    # Check if already exists
    if os.path.exists(save_path):
        return

    # Load, process and save
    raw_data = _load_scraped_data(year)
    processed_data = _process_scraped_data(raw_data)
    _save_all_matches_data(processed_data, year, "brazil")


def _get_championship_mapping() -> dict[str, str]:
    """Get championship code mapping."""
    return {
        "england": "E0",
        "france": "F1",
        "germany": "D1",
        "italy": "I1",
        "netherlands": "N1",
        "portugal": "P1",
        "spain": "SP1",
    }

def _build_football_data_url(year: int, championship: str) -> str:
    """Build URL for football-data.co.uk."""
    url_mask = "https://www.football-data.co.uk/mmz4281/{season}/{championship}.csv"
    championship_mapping = _get_championship_mapping()
    championship_code = championship_mapping[championship]

    season_start = str(year)[2:]
    season_end = str(year+1)[2:]
    season = f"{season_start}{season_end}"

    return url_mask.format(championship=championship_code, season=season)

def _download_football_data(year: int, championship: str) -> pd.DataFrame:
    """Download data from football-data.co.uk."""
    url = _build_football_data_url(year, championship)
    data = pd.read_csv(url)
    return data[["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]

def _process_football_data_co_uk_data(data: pd.DataFrame) -> dict[str, Any]:
    """Process football-data.co.uk data into standardized format."""
    all_matches = {}
    for i in range(len(data)):
        game_id = str(i+1).zfill(3)
        info = {
            "home_team": data.loc[i, "HomeTeam"],
            "away_team": data.loc[i, "AwayTeam"],
            "goals_team1": int(data.loc[i, "FTHG"]),
            "goals_team2": int(data.loc[i, "FTAG"]),
            "result": data.loc[i, "FTR"]
        }
        all_matches[game_id] = info
    return all_matches

def generate_all_matches_from_football_data_co_uk(year: int, championship: str) -> None:
    """Download, process, and save all matches data from football-data.co.uk."""
    # Download data
    data = _download_football_data(year, championship)

    # Process data
    all_matches = _process_football_data_co_uk_data(data)

    # Save data
    _save_all_matches_data(all_matches, year, championship)


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


def _determine_num_teams(total_matches: int) -> int:
    """Determine number of teams based on total matches."""
    if total_matches == 380:
        return 20
    elif total_matches == 306:
        return 18
    else:
        raise ValueError(f"Number of total matches is {total_matches}, which is not supported")

def _build_team_mapping(data: dict[str, Any], num_games: int) -> tuple[dict[str, int], list[str]]:
    """Build team name to index mapping."""
    team_to_index: dict[str, int] = {}
    team_names: list[str] = []

    for i, game_data in enumerate(data.values()):
        if i >= num_games:
            break

        for team in [game_data.get("home_team"), game_data.get("away_team")]:
            if team not in team_to_index:
                team_names.append(team)
                team_to_index[team] = len(team_names)

    return team_to_index, team_names

def _extract_game_data(data: dict[str, Any], num_games: int, team_to_index: dict[str, int]) -> tuple:
    """Extract game data for Stan input."""
    team1: list[int] = []
    team2: list[int] = []
    goals_team1_list: list[int] = []
    goals_team2_list: list[int] = []
    results: list[float] = []

    for i, game_data in enumerate(data.values()):
        if i >= num_games:
            break

        home_team = game_data.get("home_team")
        away_team = game_data.get("away_team")
        goals_team1 = game_data.get("goals_team1")
        goals_team2 = game_data.get("goals_team2")
        result = game_data.get("result")

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

    return team1, team2, goals_team1_list, goals_team2_list, results

def _create_stan_data_structures(team1: list[int], team2: list[int], goals_team1_list: list[int],
                               goals_team2_list: list[int], results: list[float],
                               team_names: list[str], num_teams: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create Bradley-Terry and Poisson data structures."""
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
    poisson_data.update({
        "goals_team1": goals_team1_list,
        "goals_team2": goals_team2_list,
    })

    return bradley_terry_data, poisson_data

def _save_stan_input_data(bradley_terry_data: dict[str, Any], poisson_data: dict[str, Any],
                         year: int, num_games: int, championship: str) -> None:
    """Save Stan input data to JSON files."""
    output_dir = get_inputs_path(championship, year)

    bradley_terry_path = os.path.join(
        output_dir, f"bradley_terry_data_{str(num_games).zfill(3)}_games.json"
    )
    poisson_path = os.path.join(
        output_dir, f"poisson_data_{str(num_games).zfill(3)}_games.json"
    )

    save_json(bradley_terry_data, bradley_terry_path)
    save_json(poisson_data, poisson_path)

def generate_real_data_stan_input(year: int, num_games: int = 380, championship: str = "brazil") -> None:
    """Generate Stan input data from matches data."""
    # Load data
    all_matches_path = os.path.join(
        get_results_path(championship, year), "all_matches.json"
    )

    data = load_json(all_matches_path)

    # Process data
    num_teams = _determine_num_teams(len(data))
    team_to_index, team_names = _build_team_mapping(data, num_games)
    team1, team2, goals_team1_list, goals_team2_list, results = _extract_game_data(data, num_games, team_to_index)

    # Create data structures
    bradley_terry_data, poisson_data = _create_stan_data_structures(
        team1, team2, goals_team1_list, goals_team2_list, results, team_names, num_teams
    )

    # Save data
    _save_stan_input_data(bradley_terry_data, poisson_data, year, num_games, championship)


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
        get_results_path(championship, year), "all_matches.json"
    )
    data = load_json(data_path)
    return data, data_path


def _determine_num_games_by_championship(championship: str) -> int:
    """Determine number of games based on championship."""
    if championship in ["brazil", "england", "italy", "spain"]:
        return 380
    else:
        return 306

def load_real_data(year: int, championship: str) -> dict[str, Any]:
    """Load the real data for a given year and number of rounds."""
    num_games = _determine_num_games_by_championship(championship)

    try:
        data_path = os.path.join(
            get_inputs_path(championship, year),
            f"poisson_data_{num_games}_games.json"
        )
        data = load_json(data_path)
    except FileNotFoundError:
        generate_real_data_stan_input(year, num_games, championship)
        data_path = os.path.join(
            get_inputs_path(championship, year),
            f"poisson_data_{num_games}_games.json"
        )
        data = load_json(data_path)
    return data


def check_results_exist(model_name: str, year: int, num_games: int, championship: str) -> bool:
    """
    Check if the results for a given model, year, and number of rounds exist.

    Args:
        model_name (str): The name of the model.
        year (int): The year of the data.
        num_games (int): The number of games of the data.
        championship (str): The championship of the data.

    Returns:
        bool: True if the results exist, False otherwise.
    """
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "real_data", "results", f"{championship}", f"{year}",
        f"{model_name}", f"{str(num_games).zfill(3)}_games"
    )
    return os.path.exists(save_dir)
