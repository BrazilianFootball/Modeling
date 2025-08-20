# pylint: disable=too-many-locals, too-many-arguments

import json
import os
import shutil
from typing import Dict, List

import cmdstanpy
import pandas as pd
import plotly.express as px
import plotly.io as pio
from constants import model_kwargs

IGNORE_COLS = [
    "chain__",
    "iter__",
    "draw__",
    "lp__",
    "accept_stat__",
    "stepsize__",
    "treedepth__",
    "n_leapfrog__",
    "divergent__",
    "energy__",
    "log_lik",
]
NUM_TEAMS = 20


def generate_real_data_stan_input(year: int, num_games: int = 380):
    """
    Load and process game data for the Bradley-Terry and Poisson models,
    then save the processed data as JSON files in the real_data directory.

    Args:
        year (int): Year of the games to load and process.
        num_games (int, optional): Number of games to process. Defaults to 380.
    """
    path = os.path.join(os.path.dirname(__file__), "../../../Data/results/processed/")
    file_path = os.path.join(path, f"Serie_A_{year}_games.json")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    team_to_index = {}
    team_names = []

    team1: List[int] = []
    team2: List[int] = []
    goals_team1_list: List[int] = []
    goals_team2_list: List[int] = []
    results = []
    for game_data in data.values():
        if len(team1) > num_games:
            break

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

    output_dir = os.path.join(os.path.dirname(__file__), "../../real_data/inputs")
    os.makedirs(output_dir, exist_ok=True)

    bradley_terry_path = os.path.join(
        output_dir, f"bradley_terry_data_{year}_{num_games}.json"
    )
    poisson_path = os.path.join(output_dir, f"poisson_data_{year}_{num_games}.json")

    with open(bradley_terry_path, "w", encoding="utf-8") as f:
        json.dump(bradley_terry_data, f, ensure_ascii=False, indent=2)
    with open(poisson_path, "w", encoding="utf-8") as f:
        json.dump(poisson_data, f, ensure_ascii=False, indent=2)

    print(f"Data saved successfully in {output_dir}.")


def run_model_with_real_data(model_name: str, year: int, num_games: int = 380):
    """
    Run the specified statistical model (Bradley-Terry or Poisson)
    using real data for a given year.

    This function loads the appropriate data file, prepares the output directories,
    compiles the corresponding Stan model, and runs the sampling procedure.
    The resulting samples are saved to disk.

    Args:
        model_name (str): The name of the model to run ("bradley_terry" or "poisson").
        year (int): The year of the real data to use.
        num_games (int, optional): Number of games to use. Defaults to 380.

    Returns:
        tuple: (fit, team_mapping, model_name_dir)
    """
    save_dir = os.path.join(os.path.dirname(__file__), "../../real_data/results")
    os.makedirs(save_dir, exist_ok=True)
    model_name_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_name_dir, exist_ok=True)

    samples_dir = os.path.join(model_name_dir, f"{year}_{num_games}_samples")
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)

    stan_model = cmdstanpy.CmdStanModel(stan_file=f"models/{model_name}.stan")
    real_data_file = "bradley_terry" if "bradley_terry" in model_name else "poisson"
    with open(
        f"real_data/inputs/{real_data_file}_data_{year}_{num_games}.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    team_mapping = {i + 1: team_name for i, team_name in enumerate(data["team_names"])}
    del data["team_names"]
    fit = stan_model.sample(data=data, **model_kwargs)
    fit.save_csvfiles(f"{model_name_dir}/{year}_{num_games}_samples")

    return fit, team_mapping, model_name_dir


def generate_boxplot(
    samples: pd.DataFrame,
    team_mapping: Dict[int, str],
    year: int,
    save_dir: str,
    model_name: str,
    num_games: int,
):
    """
    Generate a Plotly boxplot of the Bradley-Terry model's team strengths and save it as a PNG file
    in the same directory as the model results.

    Args:
        samples (pd.DataFrame): DataFrame containing the samples from the model.
        team_mapping (dict): Dictionary mapping team indices to team names.
        year (int): Year of the data.
        save_dir (str): Directory to save the boxplot.
        model_name (str): Name of the model.
        num_games (int): Number of games used in the model.

    Returns:
        fig: Plotly figure object.
    """
    column_mapping = {}
    for col in samples.columns:
        if "[" not in col:
            continue
        team_idx = int(col.split("[")[1].split("]")[0])
        if team_idx in team_mapping:
            column_mapping[col] = team_mapping[team_idx]

    if not column_mapping:
        raise ValueError("No skill columns found for teams.")

    if len(column_mapping) == NUM_TEAMS:
        samples = samples.rename(columns=column_mapping)
    elif len(column_mapping) == 2 * NUM_TEAMS:
        column_mapping = {
            from_value: to_value + " (atk)"
            if "alpha" in from_value
            else to_value + " (def)"
            for from_value, to_value in column_mapping.items()
        }
        samples = samples.rename(columns=column_mapping)
        for team in team_mapping.values():
            samples[team] = samples[team + " (atk)"] - samples[team + " (def)"]
    else:
        map_case = {
            "alpha": " (atk home)",
            "gamma": " (atk away)",
            "delta": " (def home)",
            "beta": " (def away)",
        }
        column_mapping = {
            from_value: to_value + map_case[from_value.split("[")[0]]
            for from_value, to_value in column_mapping.items()
        }
        samples = samples.rename(columns=column_mapping)
        for team in team_mapping.values():
            samples[team] = (
                samples[team + " (atk home)"] + samples[team + " (atk away)"]
            ) / 2 - (samples[team + " (def home)"] - samples[team + " (def away)"]) / 2

    samples = samples[list(team_mapping.values())]
    samples_long = samples.melt(var_name="Team", value_name="Strength")
    team_means = (
        samples_long.groupby("Team")["Strength"].mean().sort_values(ascending=False)
    )
    samples_long["Team"] = pd.Categorical(
        samples_long["Team"], categories=team_means.index, ordered=True
    )

    fig = px.box(
        samples_long,
        y="Team",
        x="Strength",
        color="Team",
        category_orders={"Team": team_means.index.tolist()},
        title=f"Strength of Teams - Serie A {year} ({num_games} games)",
        width=1000,
        height=700,
        points=False,
        template="plotly_white",
    )
    fig.update_layout(
        showlegend=False,
        font={"size": 14},
        title_font={"size": 22, "family": "Arial", "color": "black"},
        xaxis_title="Strength of Team (log-odds)",
        yaxis_title="Teams",
        margin={"l": 80, "r": 40, "t": 80, "b": 60},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(
        save_dir, f"{model_name}_team_strengths_{year}_{num_games}_boxplot.png"
    )
    pio.write_image(fig, file_path, format="png", scale=2)
    print(f"Boxplot saved as PNG in: {file_path}")


def run_real_data_model(model_name: str, year: int, num_games: int = 380):
    """
    Run the specified statistical model (Bradley-Terry or Poisson)
    using real data for a given year and generate a Plotly boxplot saved as PNG.

    Args:
        model_name (str): The name of the model to run.
        year (int): The year of the real data to use.
        num_games (int, optional): Number of games to use. Defaults to 380.
    """
    generate_real_data_stan_input(year, num_games)
    fit, team_mapping, model_save_dir = run_model_with_real_data(
        model_name, year, num_games
    )
    samples = fit.draws_pd()
    ignore_cols = [col for col in samples.columns if "raw" in col] + IGNORE_COLS
    samples = samples.drop(columns=ignore_cols)
    generate_boxplot(samples, team_mapping, year, model_save_dir, model_name, num_games)


if __name__ == "__main__":
    models = [
        "bradley_terry_3",
        "bradley_terry_4",
        "poisson_1",
        "poisson_2",
        "poisson_3",
        "poisson_4",
        "poisson_5",
    ]

    for model in models:
        run_real_data_model(model, 2024, num_games=380)
        run_real_data_model(model, 2024, num_games=50)
        run_real_data_model(model, 2024, num_games=100)
        run_real_data_model(model, 2024, num_games=150)
        run_real_data_model(model, 2024, num_games=200)
