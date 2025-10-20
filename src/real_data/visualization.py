# pylint: disable=too-many-locals, wrong-import-position, too-many-arguments, too-many-positional-arguments

import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.subplots as psub

from colors import color_mapping


def _configure_axes_optimized(fig: go.Figure, n_clubs: int) -> None:
    """
    Optimized helper function to configure axes for a multi-panel Plotly figure.

    Args:
        fig (go.Figure): The Plotly figure object to configure.
        n_clubs (int): The number of clubs (subplots) to configure axes for.

    Returns:
        None
    """
    for i in range(n_clubs):
        fig.layout.annotations[i].font.size = 8.5
        row = i // 5 + 1
        col = i % 5 + 1

        xaxis_config = {
            "row": row,
            "col": col,
            "tickfont": {"size": 7},
        }
        if row == 4:
            xaxis_config.update({
                "title_text": "Games",
                "title_font": {"size": 7.5},
            })

        fig.update_xaxes(**xaxis_config)

        yaxis_config = {
            "row": row,
            "col": col,
            "tickfont": {"size": 7},
        }
        if col == 1:
            yaxis_config.update({
                "title_text": "Points",
                "title_font": {"size": 7.5},
            })

        fig.update_yaxes(**yaxis_config)


def generate_quantiles(
    points_matrix: np.ndarray,
    current_scenario: dict[str, list[tuple[bool, int]]],
    team_mapping: dict[int, str],
    num_games: int,
    save_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates quantiles for the simulated points of each team and saves them to a CSV file.

    This function calculates specified percentiles (from 2.5 to 97.5) for the simulated points
    of each team after a given number of games. For each team, it creates a DataFrame containing
    the quantiles, the team name, the actual points, and whether the team played in each round.
    All teams' DataFrames are concatenated and saved as a CSV file in the specified directory.

    Args:
        points_matrix (np.ndarray): Array of simulated points for each team, round, and simulation.
        current_scenario (dict[str, list[tuple[bool, int]]]): Dictionary with the actual points
            evolution for each team.
        team_mapping (dict[int, str]): Mapping from team indices to team names.
        num_games (int): Number of games already played.
        save_dir (str): Directory where the CSV file will be saved.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The arrays of the 2.5th, 50th (median), and
            97.5th percentiles for each team and round
    """
    percentiles = np.linspace(2.5, 97.5, 39)
    quantiles = np.quantile(points_matrix, percentiles / 100, axis=2)
    columns = [f"p{percentile:.2f}" for percentile in percentiles]
    all_quantiles = []
    for idx, team in team_mapping.items():
        team_current_points = current_scenario[team][num_games-1][1]
        df = pd.DataFrame(
            data=team_current_points + quantiles[:, idx-1, :].T.round(3),
            columns=columns
        )
        df["team"] = team
        df["real_points"] = [point[1] for point in current_scenario[team][num_games:]]
        df["team_played"] = [point[0] for point in current_scenario[team][num_games:]]
        df.reset_index(inplace=True)
        df.rename(columns={"index": "game_id"}, inplace=True)
        df["game_id"] += num_games + 1
        all_quantiles.append(df)

    all_quantiles = pd.concat(all_quantiles)
    all_quantiles.to_csv(os.path.join(save_dir, "all_quantiles.csv"), index=False)

    return quantiles[0], quantiles[19], quantiles[-1]


def generate_points_evolution_by_team(
    points_matrix: np.ndarray,
    current_scenario: dict[str, list[tuple[bool, int]]],
    team_mapping: dict[int, str],
    num_games: int,
    save_dir: str,
    make_plots: bool = True,
) -> np.ndarray:
    """
    Generate and save a multi-panel plot showing the evolution of points for each team,
    including actual and simulated points trajectories.

    Args:
        points_matrix (np.ndarray): Simulated points for each team, round, and simulation.
        current_scenario (Dict[str, List[int]]): Actual points evolution for each team.
        team_mapping (Dict[int, str]): Mapping from team indices to team names.
        num_games (int): Number of games already played.
        save_dir (str): Directory to save the plot.
        make_plots (bool, optional): Whether to make plots. Defaults to True.

    Returns:
        np.ndarray: Final points distribution.
    """
    points_on_current_scenario = {
        team: [point[1] for point in current_scenario[team]] for team in team_mapping.values()
    }
    final_points = np.array(
        [points_on_current_scenario[team][-1] for team in team_mapping.values()]
    )
    sorted_indices = np.argsort(-final_points, kind="stable")
    sorted_team_names = [list(team_mapping.values())[i] for i in sorted_indices]

    points_at_current_round = np.array([
        points_on_current_scenario[team][num_games - 1] for team in sorted_team_names
    ])

    final_points_distribution = points_matrix[:, -1, :].copy()
    for idx, team in team_mapping.items():
        final_points_distribution[idx - 1, :] = (
            points_matrix[idx - 1, -1, :] + points_on_current_scenario[team][num_games - 1]
        )

    n_clubs = len(team_mapping)
    p2_5_points, median_points, p97_5_points = generate_quantiles(
        points_matrix, current_scenario, team_mapping, num_games, save_dir
    )

    if not make_plots:
        return final_points_distribution

    n_total_matches = n_clubs * (n_clubs - 1)
    simulation_range = np.arange(num_games + 1, n_total_matches)

    team_colors = [color_mapping.get(team, "rgba(0,0,0,1)") for team in sorted_team_names]
    team_colors_alpha = [color.replace(",1)", ",0.25)") for color in team_colors]

    fig = psub.make_subplots(
        rows=4,
        cols=5,
        subplot_titles=sorted_team_names,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
    )

    for idx, (team_idx, team_name) in enumerate(zip(sorted_indices, sorted_team_names)):
        row = idx // 5 + 1
        col = idx % 5 + 1

        med = median_points[team_idx, :] + points_at_current_round[idx]
        p2_5 = p2_5_points[team_idx, :] + points_at_current_round[idx]
        p97_5 = p97_5_points[team_idx, :] + points_at_current_round[idx]

        team_points = np.array(points_on_current_scenario[team_name])

        club_color = team_colors[idx]
        club_color_alpha = team_colors_alpha[idx]

        traces = [
            go.Scatter(
                x=simulation_range,
                y=med,
                mode="lines",
                line={"color": club_color, "width": 1},
                name="Median simulated",
                showlegend=(idx == 0),
            ),
            go.Scatter(
                x=np.concatenate([simulation_range, simulation_range[::-1]]),
                y=np.concatenate([p2_5, p97_5[::-1]]),
                fill="toself",
                fillcolor=club_color_alpha,
                line={"color": club_color_alpha, "width": 1},
                hoverinfo="skip",
                name="95% interval",
                showlegend=(idx == 0),
            ),
            go.Scatter(
                x=np.arange(1, n_total_matches),
                y=team_points,
                mode="lines",
                line={"color": "red", "dash": "dash", "width": 1},
                name="Actual",
                showlegend=(idx == 0),
            )
        ]

        for trace in traces:
            fig.add_trace(trace, row=row, col=col)

    _configure_axes_optimized(fig, n_clubs)

    fig.update_layout(
        height=900,
        width=1200,
        title_text=f"Points evolution by team (simulated after {num_games} games)",
        title_font={"size": 14, "family": "Arial"},
        showlegend=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 8},
        },
        margin={"t": 80, "b": 40},
        font={"size": 8},
    )

    file_path = os.path.join(save_dir, "points_evolution_by_team.png")
    pio.write_image(fig, file_path, format="png", scale=1, engine="kaleido")

    return final_points_distribution


def generate_boxplot(
    samples: pd.DataFrame,
    year: int,
    save_dir: str,
    num_games: int,
) -> None:
    """
    Generate a Plotly boxplot of the team strengths and save it as a PNG file
    in the specified directory.

    Args:
        samples (pd.DataFrame): DataFrame containing the samples from the model.
        year (int): Year of the data.
        save_dir (str): Directory to save the boxplot.
        num_games (int): Number of games used in the model.

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
                line_width=1,
                marker_size=3,
            )
        )

    fig.update_layout(
        height=900,
        width=1200,
        title=f"Team Strengths - Serie A {year} (after {num_games} games)",
        title_font={"size": 14, "family": "Arial"},
        xaxis_title="Team Strength (log scale)",
        yaxis_title="Teams",
        xaxis_title_font={"size": 10},
        yaxis_title_font={"size": 10},
        xaxis_tickfont={"size": 9},
        margin={"l": 80, "r": 40, "t": 80, "b": 60},
        template="plotly_white",
        showlegend=False,
        yaxis={
            "categoryorder": "array",
            "categoryarray": team_means.index.tolist(),
            "tickfont": {"size": 9},
        },
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1)

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "team_strengths_boxplot.png")
    pio.write_image(fig, file_path, format="png", scale=1, engine="kaleido")
