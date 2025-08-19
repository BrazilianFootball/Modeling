# pylint: disable=too-many-locals

import json
import os
import shutil

import cmdstanpy
from constants import model_kwargs


def load_and_save_game_data(year: int):
    """
    Loads and processes game data for Bradley-Terry and Poisson models,
    then saves the processed data as JSON files in the real_data directory.

    Args:
        year: Year of the games to load and process.
    """
    path = os.path.join(os.path.dirname(__file__), "../../../Data/results/processed/")
    file_path = os.path.join(path, f"Serie_A_{year}_games.json")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    team_to_index = {}
    team_names = []

    num_games = len(data)
    team1 = []
    team2 = []
    goals_team1_list = []
    goals_team2_list = []
    results = []
    for game_data in data.values():
        home_team = game_data.get("Home")
        away_team = game_data.get("Away")
        result = game_data.get("Result")

        if home_team not in team_to_index:
            team_names.append(home_team)
            team_to_index[home_team] = len(team_names)

        if away_team not in team_to_index:
            team_names.append(away_team)
            team_to_index[away_team] = len(team_names)

        goals_team1, goals_team2 = map(int, result.lower().split(" x "))

        team1.append(team_to_index[home_team])
        team2.append(team_to_index[away_team])
        goals_team1_list.append(goals_team1)
        goals_team2_list.append(goals_team2)

        if goals_team1 > goals_team2:
            results.append(1.0)
        elif goals_team1 < goals_team2:
            results.append(0.0)
        else:
            results.append(0.5)

    num_teams = len(team_names)

    base_data = {
        "num_games": num_games,
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

    output_dir = os.path.join(os.path.dirname(__file__), "../../real_data")
    os.makedirs(output_dir, exist_ok=True)

    bradley_terry_path = os.path.join(output_dir, f"bradley_terry_data_{year}.json")
    poisson_path = os.path.join(output_dir, f"poisson_data_{year}.json")

    with open(bradley_terry_path, "w", encoding="utf-8") as f:
        json.dump(bradley_terry_data, f, ensure_ascii=False, indent=2)
    with open(poisson_path, "w", encoding="utf-8") as f:
        json.dump(poisson_data, f, ensure_ascii=False, indent=2)

    print(f"Data saved successfully in {output_dir}.")


def run_model_with_real_data(model_name: str, year: int):
    """
    Runs the specified statistical model (Bradley-Terry or Poisson)
    using real data for a given year.

    This function loads the appropriate data file, prepares the output directories,
    compiles the corresponding Stan model, and runs the sampling procedure.
    The resulting samples are saved to disk.

    Args:
        model_name (str): The name of the model to run ("bradley_terry" or "poisson").
        year (int): The year of the real data to use.

    Returns:
        None
    """
    save_dir = os.path.join(os.path.dirname(__file__), "../../real_data/results")
    os.makedirs(save_dir, exist_ok=True)
    model_name_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_name_dir, exist_ok=True)

    samples_dir = os.path.join(model_name_dir, f"{year}_samples")
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)

    model_name_stan = (
        "bradley_terry_4" if model_name == "bradley_terry" else "poisson_5"
    )
    model = cmdstanpy.CmdStanModel(stan_file=f"models/{model_name_stan}.stan")
    with open(f"real_data/{model_name}_data_{year}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    del data["team_names"]
    fit = model.sample(data=data, **model_kwargs)
    fit.save_csvfiles(f"{model_name_dir}/{year}_samples")

    return fit


if __name__ == "__main__":
    for season in range(2013, 2025):
        print(f"Processing year {season}")
        load_and_save_game_data(season)
        fit_bradley_terry = run_model_with_real_data("bradley_terry", season)
        fit_poisson = run_model_with_real_data("poisson", season)
