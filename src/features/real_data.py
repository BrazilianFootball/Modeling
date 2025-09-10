# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-positional-arguments

import json
import logging
import os
import shutil
from itertools import product
from typing import Any

import cmdstanpy
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.subplots as psub
from colors import color_mapping
from constants import model_kwargs
from generators import simulate_bradley_terry

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

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


def generate_real_data_stan_input(year: int, num_rounds: int = 38) -> None:
    """
    Load and process real Serie A game data for a given year and number of rounds,
    and save the processed data as JSON files for use with the Bradley-Terry and Poisson models.

    Args:
        year (int): The year of the games to load and process.
        num_rounds (int, optional): Number of rounds to process. Defaults to 38.
    """
    path = os.path.join(os.path.dirname(__file__), "../../../Data/results/processed/")
    file_path = os.path.join(path, f"Serie_A_{year}_games.json")

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    team_to_index: dict[str, int] = {}
    team_names: list[str] = []

    team1: list[int] = []
    team2: list[int] = []
    goals_team1_list: list[int] = []
    goals_team2_list: list[int] = []
    results: list[float] = []
    for game_data in data.values():
        if len(team1) > num_rounds * 10:
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
        output_dir, f"bradley_terry_data_{year}_{num_rounds}.json"
    )
    poisson_path = os.path.join(output_dir, f"poisson_data_{year}_{num_rounds}.json")

    with open(bradley_terry_path, "w", encoding="utf-8") as f:
        json.dump(bradley_terry_data, f, ensure_ascii=False, indent=2)
    with open(poisson_path, "w", encoding="utf-8") as f:
        json.dump(poisson_data, f, ensure_ascii=False, indent=2)

    print(f"Data saved successfully in {output_dir}.")


def generate_all_matches_data(year: int) -> None:
    """
    Generate all matches data for a given year.

    Args:
        year (int): The year of the data to generate.
    """
    save_dir = os.path.join(os.path.dirname(__file__), "../../real_data/results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"all_matches_{year}.json")
    if os.path.exists(save_path):
        return

    path = os.path.join(os.path.dirname(__file__), "../../../Data/results/processed/")
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

    print(f"All matches data saved successfully in {save_path}.")


def update_probabilities(
    probabilities: dict[str, Any], year: int, model_name: str, num_rounds: int
) -> None:
    """
    Update the probabilities for a given year, model name, and number of rounds.

    Args:
        probabilities (dict[str, Any]): The probabilities to update.
        year (int): The year of the data to update.
        model_name (str): The name of the model to update.
        num_rounds (int): The number of rounds to update.
    """
    data_path = os.path.join(
        os.path.dirname(__file__), f"../../real_data/results/all_matches_{year}.json"
    )
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    for game_id, probabilities_data in probabilities.items():
        assert data[game_id]["home_team"] == probabilities_data["home_team"]
        assert data[game_id]["away_team"] == probabilities_data["away_team"]
        data[game_id]["probabilities"] = data[game_id].get("probabilities", {})
        data[game_id]["probabilities"][model_name] = data[game_id]["probabilities"].get(
            model_name, {}
        )
        data[game_id]["probabilities"][model_name][
            str(num_rounds)
        ] = probabilities_data["probabilities"]

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"Probabilities updated successfully in {data_path} for model {model_name}"
        f" with {num_rounds} rounds on {year}."
    )


def run_model_with_real_data(
    model_name: str, year: int, num_rounds: int = 38
) -> tuple[cmdstanpy.CmdStanMCMC, dict[int, str], str]:
    """
    Run the specified statistical model (Bradley-Terry or Poisson) using real data
    for a given year and number of rounds. Loads the appropriate data file, prepares
    output directories, compiles the Stan model, and runs sampling. Saves the resulting
    samples to disk.

    Args:
        model_name (str): The name of the model to run ("bradley_terry" or "poisson").
        year (int): The year of the real data to use.
        num_rounds (int, optional): Number of rounds to use on fit. Defaults to 38.

    Returns:
        Tuple[cmdstanpy.CmdStanMCMC, Dict[int, str], str]:
            fit: The CmdStanPy fit object.
            team_mapping: Dictionary mapping team indices to team names.
            model_name_dir: Directory where model results are saved.
    """
    save_dir = os.path.join(os.path.dirname(__file__), "../../real_data/results")
    os.makedirs(save_dir, exist_ok=True)
    model_name_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_name_dir, exist_ok=True)

    samples_dir = os.path.join(model_name_dir, f"{year}_{num_rounds}")
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)

    stan_model = cmdstanpy.CmdStanModel(stan_file=f"models/{model_name}.stan")
    real_data_file = "bradley_terry" if "bradley_terry" in model_name else "poisson"
    with open(
        f"real_data/inputs/{real_data_file}_data_{year}_{num_rounds}.json",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    team_mapping: dict[int, str] = {
        i + 1: team_name for i, team_name in enumerate(data["team_names"])
    }
    del data["team_names"]
    fit = stan_model.sample(data=data, **model_kwargs)
    fit.save_csvfiles(samples_dir)

    return fit, team_mapping, samples_dir


def set_team_strengths(
    samples: pd.DataFrame, team_mapping: dict[int, str]
) -> pd.DataFrame:
    """
    Rename and process the columns of the samples DataFrame to map team indices to team names,
    and compute the overall team strengths depending on the model structure.

    Args:
        samples (pd.DataFrame): DataFrame containing the samples from the model.
        team_mapping (Dict[int, str]): Mapping from team indices to team names.

    Returns:
        pd.DataFrame: DataFrame with columns renamed to team names and team strengths computed.
    """
    column_mapping: dict[str, str] = {}
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
    return samples


def generate_boxplot(
    samples: pd.DataFrame,
    year: int,
    save_dir: str,
    num_rounds: int,
) -> None:
    """
    Generate a Plotly boxplot of the team strengths and save it as a PNG file
    in the specified directory.

    Args:
        samples (pd.DataFrame): DataFrame containing the samples from the model.
        year (int): Year of the data.
        save_dir (str): Directory to save the boxplot.
        num_rounds (int): Number of rounds used in the model.

    Returns:
        None
    """
    samples_long = samples.melt(var_name="Team", value_name="Strength")
    team_means = (
        samples_long.groupby("Team")["Strength"].mean().sort_values(ascending=True)
    )
    samples_long["Team"] = pd.Categorical(
        samples_long["Team"], categories=team_means.index, ordered=True
    )

    fig = go.Figure()
    for team in team_means.index:
        team_data = samples_long[samples_long["Team"] == team]
        cor = color_mapping.get(team, "rgba(0,0,0,1)")
        fig.add_trace(
            go.Box(
                x=team_data["Strength"],
                y=team_data["Team"],
                name=team,
                marker_color=cor,
                boxmean=False,
                orientation="h",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"Team Strengths - Serie A {year} ({num_rounds} rounds)",
        width=1000,
        height=700,
        font={"size": 14},
        title_font={"size": 22, "family": "Arial", "color": "black"},
        xaxis_title="Team Strength (log scale)",
        yaxis_title="Teams",
        margin={"l": 80, "r": 40, "t": 80, "b": 60},
        template="plotly_white",
        showlegend=False,
        yaxis={"categoryorder": "array", "categoryarray": team_means.index.tolist()},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "team_strengths_boxplot.png")
    pio.write_image(fig, file_path, format="png", scale=2)
    print(f"Boxplot saved as PNG in: {file_path}")


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
            f"real_data/inputs/poisson_data_{year}_38.json",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
    except FileNotFoundError:
        generate_real_data_stan_input(year, 38)
        with open(
            f"real_data/inputs/poisson_data_{year}_38.json",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
    return data


def get_real_points_evolution(
    data: dict[str, Any], team_mapping: dict[int, str]
) -> dict[str, list[int]]:
    """
    Calculate the current points evolution for each team based on real results.

    Args:
        data (Dict[str, Any]): Real data loaded.
        team_mapping (Dict[int, str]): Mapping from team indices to names.

    Returns:
        Dict[str, List[int]]: Dictionary with the points evolution for each team.
    """
    home_team = data["team1"]
    away_team = data["team2"]
    home_team_names = [team_mapping[team] for team in home_team]
    away_team_names = [team_mapping[team] for team in away_team]

    current_scenario: dict[str, list[int]] = {
        team: [] for team in team_mapping.values()
    }
    accumulated_points: dict[str, int] = dict.fromkeys(team_mapping.values(), 0)

    for i, (home, away) in enumerate(
        zip(home_team_names, away_team_names, strict=False)
    ):
        goals_home = data["goals_team1"][i]
        goals_away = data["goals_team2"][i]
        if goals_home > goals_away:
            points_home = 3
            points_away = 0
        elif goals_home < goals_away:
            points_home = 0
            points_away = 3
        else:
            points_home = 1
            points_away = 1

        accumulated_points[home] += points_home
        accumulated_points[away] += points_away

        current_scenario[home].append(accumulated_points[home])
        current_scenario[away].append(accumulated_points[away])

    return current_scenario


def generate_points_matrix_bradley_terry(
    samples: pd.DataFrame,
    team_mapping: dict[int, str],
    data: dict[str, Any],
    num_rounds: int,
    num_simulations: int,
    n_matches_per_club: int,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """
    Generate a points matrix for the remainder of the season using the Bradley-Terry model.

    Args:
        samples (pd.DataFrame): Posterior samples of team strengths.
        team_mapping (Dict[int, str]): Mapping from team indices to names.
        data (Dict[str, Any]): Real data loaded.
        num_rounds (int): Number of rounds already played.
        num_simulations (int): Number of simulations to run.
        n_matches_per_club (int): Number of matches per club to simulate.

    Returns:
        np.ndarray: Points matrix of shape (n_teams, n_matches_per_club, num_simulations).
    """
    home_team = data["team1"]
    away_team = data["team2"]
    home_team_names = [team_mapping[team] for team in home_team]
    away_team_names = [team_mapping[team] for team in away_team]
    teams = list(team_mapping.values())
    n_teams = len(teams)
    points_matrix = np.zeros((n_teams, n_matches_per_club, num_simulations), dtype=int)
    samples_indices = np.random.randint(
        0, len(samples), size=(n_matches_per_club, num_simulations)
    )

    probabilities: dict[str, dict[str, Any]] = {}
    for rd in range(n_matches_per_club):
        for game in range(10):
            game_id = (num_rounds + rd) * 10 + game
            home_strengths = samples.iloc[samples_indices[rd]][
                home_team_names[game_id]
            ].values
            away_strengths = samples.iloc[samples_indices[rd]][
                away_team_names[game_id]
            ].values
            if "kappa" in samples.columns:
                kappa_values = samples.iloc[samples_indices[rd]]["kappa"].values
            else:
                kappa_values = np.zeros(num_simulations)

            results = simulate_bradley_terry(
                home_strengths, away_strengths, kappa_values
            )
            home_idx = home_team[game_id] - 1
            away_idx = away_team[game_id] - 1
            home_new_points = (results == 1) * 3 + (results == 0.5) * 1
            away_new_points = (results == 0) * 3 + (results == 0.5) * 1
            if rd > 0:
                points_matrix[home_idx, rd, :] = (
                    points_matrix[home_idx, rd - 1, :] + home_new_points
                )
                points_matrix[away_idx, rd, :] = (
                    points_matrix[away_idx, rd - 1, :] + away_new_points
                )
            else:
                points_matrix[home_idx, rd, :] = home_new_points
                points_matrix[away_idx, rd, :] = away_new_points

            probs = [
                float(np.sum(results == 1) / num_simulations),
                float(np.sum(results == 0.5) / num_simulations),
                float(np.sum(results == 0) / num_simulations),
            ]
            probabilities[str(game_id + 1).zfill(3)] = {}
            probabilities[str(game_id + 1).zfill(3)]["home_team"] = home_team_names[
                game_id
            ]
            probabilities[str(game_id + 1).zfill(3)]["away_team"] = away_team_names[
                game_id
            ]
            probabilities[str(game_id + 1).zfill(3)]["probabilities"] = probs
    return points_matrix, probabilities


def generate_points_matrix_poisson(
    samples: pd.DataFrame,
    team_mapping: dict[int, str],
    data: dict[str, Any],
    num_rounds: int,
    num_simulations: int,
    n_matches_per_club: int,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """
    Generate a points matrix for the remainder of the season using the Poisson model.

    Args:
        samples (pd.DataFrame): Posterior samples of team strengths.
        team_mapping (Dict[int, str]): Mapping from team indices to names.
        data (Dict[str, Any]): Real data loaded.
        num_rounds (int): Number of rounds already played.
        num_simulations (int): Number of simulations to run.
        n_matches_per_club (int): Number of matches per club to simulate.

    Returns:
        np.ndarray: Points matrix of shape (n_teams, n_matches_per_club, num_simulations).
    """
    home_team = data["team1"]
    away_team = data["team2"]
    home_team_names = [team_mapping[team] for team in home_team]
    away_team_names = [team_mapping[team] for team in away_team]
    teams = list(team_mapping.values())
    n_teams = len(teams)
    points_matrix = np.zeros((n_teams, n_matches_per_club, num_simulations), dtype=int)
    samples_indices = np.random.randint(
        0, len(samples), size=(n_matches_per_club, num_simulations)
    )

    probabilities: dict[str, dict[str, Any]] = {}
    for rd in range(n_matches_per_club):
        if "nu" in samples.columns:
            nu = samples.iloc[samples_indices[rd]]["nu"].values
        else:
            nu = np.zeros(num_simulations)
        for game in range(10):
            game_id = (num_rounds + rd) * 10 + game
            home_name = home_team_names[game_id]
            away_name = away_team_names[game_id]
            if home_name + " (atk home)" in samples.columns:
                atk_home = samples.iloc[samples_indices[rd]][
                    home_name + " (atk home)"
                ].values
                def_away = samples.iloc[samples_indices[rd]][
                    away_name + " (def away)"
                ].values
                atk_away = samples.iloc[samples_indices[rd]][
                    away_name + " (atk away)"
                ].values
                def_home = samples.iloc[samples_indices[rd]][
                    home_name + " (def home)"
                ].values
                home_strengths = atk_home + def_away
                away_strengths = atk_away + def_home
            elif home_name + " (atk)" in samples.columns:
                atk_strength = samples.iloc[samples_indices[rd]][
                    home_name + " (atk)"
                ].values
                def_strength = samples.iloc[samples_indices[rd]][
                    away_name + " (def)"
                ].values
                home_strengths = atk_strength + def_strength
                away_strengths = atk_strength + def_strength
            else:
                home_strengths = samples.iloc[samples_indices[rd]][home_name].values
                away_strengths = samples.iloc[samples_indices[rd]][away_name].values

            home_strengths = np.exp(home_strengths + nu)
            away_strengths = np.exp(away_strengths + nu)

            home_goals = np.random.poisson(home_strengths)
            away_goals = np.random.poisson(away_strengths)

            home_win = home_goals > away_goals
            away_win = home_goals < away_goals
            tie = home_goals == away_goals

            home_idx = home_team[game_id] - 1
            away_idx = away_team[game_id] - 1
            if rd > 0:
                points_matrix[home_idx, rd, :] = (
                    points_matrix[home_idx, rd - 1, :] + home_win * 3 + tie * 1
                )
                points_matrix[away_idx, rd, :] = (
                    points_matrix[away_idx, rd - 1, :] + away_win * 3 + tie * 1
                )
            else:
                points_matrix[home_idx, rd, :] = home_win * 3 + tie * 1
                points_matrix[away_idx, rd, :] = away_win * 3 + tie * 1

            probs = [
                float(np.sum(home_win) / num_simulations),
                float(np.sum(tie) / num_simulations),
                float(np.sum(away_win) / num_simulations),
            ]
            probabilities[str(game_id + 1).zfill(3)] = {}
            probabilities[str(game_id + 1).zfill(3)]["home_team"] = home_name
            probabilities[str(game_id + 1).zfill(3)]["away_team"] = away_name
            probabilities[str(game_id + 1).zfill(3)]["probabilities"] = probs
    return points_matrix, probabilities


def simulate_competition(
    samples: pd.DataFrame,
    team_mapping: dict[int, str],
    model_name: str,
    year: int,
    num_rounds: int,
    num_simulations: int = 1000,
) -> tuple[np.ndarray, dict[str, list[int]], dict[str, dict[str, Any]]]:
    """
    Simulate the remainder of the Serie A season using posterior samples from the model,
    generating possible points trajectories for each team.

    Args:
        samples (pd.DataFrame): DataFrame with model samples.
        team_mapping (Dict[int, str]): Mapping from team indices to names.
        model_name (str): Model name.
        year (int): Data year.
        num_rounds (int): Number of rounds already played.
        num_simulations (int, optional): Number of simulations. Default: 1000.

    Returns:
        Tuple[np.ndarray, Dict[str, List[int]]]:
            points_matrix: Array of simulated points for each team, round, and simulation.
            current_scenario: Dictionary with the actual points evolution for each team.
    """
    data = load_real_data(year)
    n_matches_per_club = 38 - num_rounds

    current_scenario = get_real_points_evolution(data, team_mapping)

    if "bradley_terry" in model_name:
        points_matrix, probabilities = generate_points_matrix_bradley_terry(
            samples, team_mapping, data, num_rounds, num_simulations, n_matches_per_club
        )
    else:
        points_matrix, probabilities = generate_points_matrix_poisson(
            samples, team_mapping, data, num_rounds, num_simulations, n_matches_per_club
        )

    return points_matrix, current_scenario, probabilities


def generate_points_evolution_by_team(
    points_matrix: np.ndarray,
    current_scenario: dict[str, list[int]],
    team_mapping: dict[int, str],
    num_rounds: int,
    save_dir: str,
) -> None:
    """
    Generate and save a multi-panel plot showing the evolution of points for each team,
    including actual and simulated points trajectories.

    Args:
        points_matrix (np.ndarray): Simulated points for each team, round, and simulation.
        current_scenario (Dict[str, List[int]]): Actual points evolution for each team.
        team_mapping (Dict[int, str]): Mapping from team indices to team names.
        num_rounds (int): Number of rounds already played.
        save_dir (str): Directory to save the plot.

    Returns:
        None
    """
    n_matches_per_club = 38 - num_rounds
    median_points = np.median(points_matrix, axis=2)
    p95_points = np.percentile(points_matrix, 95, axis=2)
    p5_points = np.percentile(points_matrix, 5, axis=2)

    final_points = np.array(
        [current_scenario[team][-1] for team in team_mapping.values()]
    )
    sorted_indices = np.argsort(-final_points)
    sorted_team_names = [list(team_mapping.values())[i] for i in sorted_indices]

    n_rounds = median_points.shape[1]
    last_rounds = num_rounds + np.arange(n_rounds - n_matches_per_club, n_rounds) + 1

    fig = psub.make_subplots(
        rows=4,
        cols=5,
        subplot_titles=[
            [k for k, v in team_mapping.items() if v == team_idx][0]
            if [k for k, v in team_mapping.items() if v == team_idx]
            else sorted_team_names[idx]
            for idx, team_idx in enumerate(sorted_indices)
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
    )

    for idx, team_idx in enumerate(sorted_indices):
        row = idx // 5 + 1
        col = idx % 5 + 1

        club_name = sorted_team_names[idx]
        club_color = color_mapping[club_name]

        points_at_current_round = current_scenario[club_name][num_rounds - 1]
        med = median_points[team_idx, :] + points_at_current_round
        p95 = p95_points[team_idx, :] + points_at_current_round
        p5 = p5_points[team_idx, :] + points_at_current_round
        team_points = current_scenario[club_name]
        fig.add_trace(
            go.Scatter(
                x=last_rounds,
                y=med,
                mode="lines",
                line={"color": club_color},
                name="Median simulated",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

        club_color = club_color.replace(",1)", ",0.25)")
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([last_rounds, last_rounds[::-1]]),
                y=np.concatenate([p5, p95[::-1]]),
                fill="toself",
                fillcolor=club_color,
                line={"color": club_color},
                hoverinfo="skip",
                name="90% interval",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(1, 39),
                y=team_points,
                mode="lines",
                line={"color": "red", "dash": "dash", "width": 1.5},
                name="Actual",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

    for i in range(20):
        row = i // 5 + 1
        col = i % 5 + 1
        if row == 4:
            fig.update_xaxes(title_text="Rounds", row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text="Points", row=row, col=col)

    fig.update_layout(
        height=900,
        width=1200,
        title_text="Points evolution by team",
        showlegend=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"t": 80, "b": 40},
    )

    file_path = os.path.join(save_dir, "points_evolution_by_team.png")
    pio.write_image(fig, file_path, format="png", scale=2)


def run_real_data_model(
    model_name: str, year: int, num_rounds: int = 38, num_simulations: int = 1000
) -> None:
    """
    Run the specified statistical model (Bradley-Terry or Poisson) using real data
    for a given year and number of rounds, generate a boxplot of team strengths,
    and, if not all rounds are played, simulate the remainder of the season and
    generate a points evolution plot.

    Args:
        model_name (str): The name of the model to run.
        year (int): The year of the real data to use.
        num_rounds (int, optional): Number of rounds already played. Defaults to 38.
        num_simulations (int, optional): Number of simulations to run. Defaults to 1000.

    Returns:
        None
    """
    generate_all_matches_data(year)
    generate_real_data_stan_input(year, num_rounds)
    fit, team_mapping, model_save_dir = run_model_with_real_data(
        model_name, year, num_rounds
    )
    samples = fit.draws_pd()
    ignore_cols = [col for col in samples.columns if "raw" in col] + IGNORE_COLS
    samples = samples.drop(columns=ignore_cols)
    samples = set_team_strengths(samples, team_mapping)
    generate_boxplot(
        samples[list(team_mapping.values())],
        year,
        model_save_dir,
        num_rounds,
    )
    if num_rounds != 38:
        points_matrix, current_scenario, probabilities = simulate_competition(
            samples, team_mapping, model_name, year, num_rounds, num_simulations
        )
        update_probabilities(probabilities, year, model_name, num_rounds)
        generate_points_evolution_by_team(
            points_matrix,
            current_scenario,
            team_mapping,
            num_rounds,
            save_dir=model_save_dir,
        )


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

    seasons = [*range(2019, 2025)]
    rounds = [5, 10, 15, 19, 20]

    for model, season, actual_round in product(models, seasons, rounds):
        run_real_data_model(
            model, season, num_rounds=actual_round, num_simulations=100_000
        )
