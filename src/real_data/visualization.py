# pylint: disable=too-many-locals, wrong-import-position

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.subplots as psub

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from features.colors import color_mapping  # noqa: E402


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
