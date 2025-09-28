# pylint: disable=too-many-locals, wrong-import-position

import os

import numba as nb
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.subplots as psub

from colors import color_mapping

@nb.jit(nopython=True, parallel=True)
def _calculate_quantiles_fast(points_matrix: np.ndarray) -> np.ndarray:
    """
    Optimized version with Numba for quantile calculation.
    Uses a quick selection algorithm instead of full sorting.
    """
    n_teams, n_games, n_sims = points_matrix.shape
    quantiles = np.zeros((3, n_teams, n_games))

    percentiles = np.array([5.0, 50.0, 95.0])

    for i in range(n_teams):
        for j in range(n_games):
            data = points_matrix[i, j, :].copy()

            for k, p in enumerate(percentiles):
                idx = int((p / 100.0) * (n_sims - 1))
                quantiles[k, i, j] = np.partition(data, idx)[idx]

    return quantiles


# run the function to compile it
_calculate_quantiles_fast(np.random.rand(100, 100, 1000))


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


def generate_points_evolution_by_team(
    points_matrix: np.ndarray,
    current_scenario: dict[str, list[int]],
    team_mapping: dict[int, str],
    num_games: int,
    save_dir: str,
) -> None:
    """
    Generate and save a multi-panel plot showing the evolution of points for each team,
    including actual and simulated points trajectories.

    Args:
        points_matrix (np.ndarray): Simulated points for each team, round, and simulation.
        current_scenario (Dict[str, List[int]]): Actual points evolution for each team.
        team_mapping (Dict[int, str]): Mapping from team indices to team names.
        num_games (int): Number of games already played.
        save_dir (str): Directory to save the plot.

    Returns:
        None
    """
    n_clubs = len(team_mapping)
    quantiles = _calculate_quantiles_fast(points_matrix)
    # quantiles = np.quantile(points_matrix, [0.05, 0.5, 0.95], axis=2)
    p5_points = quantiles[0]
    median_points = quantiles[1]
    p95_points = quantiles[2]

    final_points = np.array(
        [current_scenario[team][-1] for team in team_mapping.values()]
    )
    sorted_indices = np.argsort(-final_points, kind="stable")
    sorted_team_names = [list(team_mapping.values())[i] for i in sorted_indices]

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

    points_at_current_round = np.array([
        current_scenario[team][num_games - 1] for team in sorted_team_names
    ])
    for idx, (team_idx, team_name) in enumerate(zip(sorted_indices, sorted_team_names)):
        row = idx // 5 + 1
        col = idx % 5 + 1

        med = median_points[team_idx, :] + points_at_current_round[idx]
        p95 = p95_points[team_idx, :] + points_at_current_round[idx]
        p5 = p5_points[team_idx, :] + points_at_current_round[idx]

        team_points = np.array(current_scenario[team_name])

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
                y=np.concatenate([p5, p95[::-1]]),
                fill="toself",
                fillcolor=club_color_alpha,
                line={"color": club_color_alpha, "width": 1},
                hoverinfo="skip",
                name="90% interval",
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
