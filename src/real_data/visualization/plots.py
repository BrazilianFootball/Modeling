# pylint: disable=too-many-locals, wrong-import-position

import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.subplots as psub

from .colors import color_mapping
from ..utils.io_utils import save_csv


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


def _calculate_percentiles(points_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate percentiles for points matrix."""
    percentiles = np.linspace(2.5, 97.5, 39)
    quantiles = np.quantile(points_matrix, percentiles / 100, axis=2)
    return percentiles, quantiles, quantiles[0], quantiles[19], quantiles[-1]

def _create_team_dataframe(team: str, team_idx: int, quantiles: np.ndarray,
                          current_scenario: dict[str, list[tuple[bool, int]]],
                          num_games: int, percentiles: np.ndarray) -> pd.DataFrame:
    """Create DataFrame for a single team's quantile data."""
    team_current_points = current_scenario[team][num_games-1][1]
    columns = [f"p{percentile:.2f}" for percentile in percentiles]

    df = pd.DataFrame(
        data=team_current_points + quantiles[:, team_idx-1, :].T.round(3),
        columns=columns
    )
    df["team"] = team
    df["real_points"] = [point[1] for point in current_scenario[team][num_games:]]
    df["team_played"] = [point[0] for point in current_scenario[team][num_games:]]
    df.reset_index(inplace=True)
    df.rename(columns={"index": "game_id"}, inplace=True)
    df["game_id"] += num_games + 1

    return df

def _save_quantiles_csv(all_quantiles: pd.DataFrame, save_dir: str) -> None:
    """Save quantiles data to CSV file."""
    save_csv(all_quantiles, os.path.join(save_dir, "all_quantiles.csv"))

def generate_quantiles(
    points_matrix: np.ndarray,
    current_scenario: dict[str, list[tuple[bool, int]]],
    team_mapping: dict[int, str],
    num_games: int,
    save_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate quantiles for simulated points and save to CSV."""
    # Calculate percentiles
    percentiles, quantiles, p2_5_points, median_points, p97_5_points = _calculate_percentiles(points_matrix)

    # Create DataFrames for each team
    all_quantiles = []
    for idx, team in team_mapping.items():
        team_df = _create_team_dataframe(team, idx, quantiles, current_scenario, num_games, percentiles)
        all_quantiles.append(team_df)

    # Combine and save
    all_quantiles_df = pd.concat(all_quantiles)
    _save_quantiles_csv(all_quantiles_df, save_dir)

    return p2_5_points, median_points, p97_5_points


def _extract_current_points_data(current_scenario: dict[str, list[tuple[bool, int]]],
                                team_mapping: dict[int, str]) -> dict[str, list[int]]:
    """Extract current points data from scenario."""
    return {
        team: [point[1] for point in current_scenario[team]]
        for team in team_mapping.values()
    }

def _sort_teams_by_final_points(points_on_current_scenario: dict[str, list[int]],
                               team_mapping: dict[int, str]) -> tuple[np.ndarray, list[str]]:
    """Sort teams by their final points."""
    final_points = np.array([
        points_on_current_scenario[team][-1] for team in team_mapping.values()
    ])
    sorted_indices = np.argsort(-final_points, kind="stable")
    sorted_team_names = [list(team_mapping.values())[i] for i in sorted_indices]
    return sorted_indices, sorted_team_names

def _prepare_team_colors(sorted_team_names: list[str]) -> tuple[list[str], list[str]]:
    """Prepare team colors for visualization."""
    team_colors = [color_mapping.get(team, "rgba(0,0,0,1)") for team in sorted_team_names]
    team_colors_alpha = [color.replace(",1)", ",0.25)") for color in team_colors]
    return team_colors, team_colors_alpha

def _create_subplot_figure(sorted_team_names: list[str]) -> go.Figure:
    """Create subplot figure for team evolution."""
    return psub.make_subplots(
        rows=4,
        cols=5,
        subplot_titles=sorted_team_names,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
    )

def _calculate_final_points_distribution(points_matrix: np.ndarray,
                                       points_on_current_scenario: dict[str, list[int]],
                                       team_mapping: dict[int, str], num_games: int) -> np.ndarray:
    """Calculate final points distribution for all teams."""
    final_points_distribution = points_matrix[:, -1, :].copy()
    for idx, team in team_mapping.items():
        final_points_distribution[idx - 1, :] = (
            points_matrix[idx - 1, -1, :] + points_on_current_scenario[team][num_games - 1]
        )
    return final_points_distribution

def _create_team_traces(team_idx: int, team_name: str, median_points: np.ndarray,
                       p2_5_points: np.ndarray, p97_5_points: np.ndarray,
                       points_at_current_round: np.ndarray, simulation_range: np.ndarray,
                       points_on_current_scenario: dict[str, list[int]], n_total_matches: int,
                       club_color: str, club_color_alpha: str, idx: int) -> list[go.Scatter]:
    """Create traces for a single team."""
    med = median_points[team_idx, :] + points_at_current_round[idx]
    p2_5 = p2_5_points[team_idx, :] + points_at_current_round[idx]
    p97_5 = p97_5_points[team_idx, :] + points_at_current_round[idx]
    team_points = np.array(points_on_current_scenario[team_name])

    return [
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

def _configure_figure_layout(fig: go.Figure, num_games: int) -> None:
    """Configure the final layout of the figure."""
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

def _save_evolution_plot(fig: go.Figure, save_dir: str) -> None:
    """Save the points evolution plot to file."""
    file_path = os.path.join(save_dir, "points_evolution_by_team.png")
    pio.write_image(fig, file_path, format="png", scale=1, engine="kaleido")

def generate_points_evolution_by_team(
    points_matrix: np.ndarray,
    current_scenario: dict[str, list[tuple[bool, int]]],
    team_mapping: dict[int, str],
    num_games: int,
    save_dir: str,
) -> np.ndarray:
    """Generate and save a multi-panel plot showing the evolution of points for each team."""
    # Generate quantiles
    p2_5_points, median_points, p97_5_points = generate_quantiles(
        points_matrix, current_scenario, team_mapping, num_games, save_dir
    )

    # Extract and process data
    points_on_current_scenario = _extract_current_points_data(current_scenario, team_mapping)
    sorted_indices, sorted_team_names = _sort_teams_by_final_points(points_on_current_scenario, team_mapping)
    team_colors, team_colors_alpha = _prepare_team_colors(sorted_team_names)

    # Create figure
    fig = _create_subplot_figure(sorted_team_names)
    n_clubs = len(team_mapping)
    n_total_matches = n_clubs * (n_clubs - 1)
    simulation_range = np.arange(num_games + 1, n_total_matches)

    # Calculate points at current round
    points_at_current_round = np.array([
        points_on_current_scenario[team][num_games - 1] for team in sorted_team_names
    ])

    # Calculate final points distribution
    final_points_distribution = _calculate_final_points_distribution(
        points_matrix, points_on_current_scenario, team_mapping, num_games
    )

    # Add traces for each team
    for idx, (team_idx, team_name) in enumerate(zip(sorted_indices, sorted_team_names)):
        row = idx // 5 + 1
        col = idx % 5 + 1

        traces = _create_team_traces(
            team_idx, team_name, median_points, p2_5_points, p97_5_points,
            points_at_current_round, simulation_range, points_on_current_scenario,
            n_total_matches, team_colors[idx], team_colors_alpha[idx], idx
        )

        for trace in traces:
            fig.add_trace(trace, row=row, col=col)

    # Configure and save
    _configure_axes_optimized(fig, n_clubs)
    _configure_figure_layout(fig, num_games)
    _save_evolution_plot(fig, save_dir)

    return final_points_distribution


def _prepare_samples_data(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare samples data for boxplot."""
    samples_long = samples.melt(var_name="Team", value_name="Strength")
    team_means = (
        samples_long.groupby("Team")["Strength"].mean().sort_values(ascending=True)
    )
    samples_long["Team"] = pd.Categorical(
        samples_long["Team"], categories=team_means.index, ordered=True
    )
    return samples_long, team_means

def _create_boxplot_figure(samples_long: pd.DataFrame, team_means: pd.Series) -> go.Figure:
    """Create boxplot figure with team data."""
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
    return fig

def _configure_boxplot_layout(fig: go.Figure, year: int, num_games: int, team_means: pd.Series) -> None:
    """Configure boxplot layout."""
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

def _save_boxplot(fig: go.Figure, save_dir: str) -> None:
    """Save boxplot to file."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "team_strengths_boxplot.png")
    pio.write_image(fig, file_path, format="png", scale=1, engine="kaleido")

def generate_boxplot(
    samples: pd.DataFrame,
    year: int,
    save_dir: str,
    num_games: int,
) -> None:
    """Generate a Plotly boxplot of the team strengths and save it as a PNG file."""
    # Prepare data
    samples_long, team_means = _prepare_samples_data(samples)

    # Create figure
    fig = _create_boxplot_figure(samples_long, team_means)

    # Configure and save
    _configure_boxplot_layout(fig, year, num_games, team_means)
    _save_boxplot(fig, save_dir)
