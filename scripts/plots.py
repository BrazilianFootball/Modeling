# Based on https://github.com/hyunjimoon/SBC/blob/master/R/plot.R
# pylint: disable=too-many-arguments

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from calculate import adjust_gamma_optimize, ecdf_intervals
from plotly.subplots import make_subplots


def _calculate_ci(
    N: int, K: int, prob: float
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Calculate confidence intervals for ECDF."""
    gamma = adjust_gamma_optimize(N, K, conf_level=prob)
    z = np.linspace(0, 1, K + 1)
    z_plot = np.concatenate([np.repeat(z[:-1], 2), [1]])

    intervals = ecdf_intervals(N, L=1, K=K, gamma=gamma)
    intervals["upper"] = np.append(intervals["upper"], [1])
    intervals["lower"] = np.append(intervals["lower"], [1])

    return z_plot, intervals


def _add_ecdf_traces(
    fig: go.Figure,
    ranks: np.ndarray,
    series_names: List[str],
    colors: Optional[List[str]] = None,
    row: int = 1,
    col: int = 1,
    is_diff: bool = False,
    showlegend: bool = True,
) -> None:
    """Add ECDF or ECDF difference traces to the plot."""
    N, M = ranks.shape

    if colors is None:
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    for i in range(M):
        sorted_ranks = np.sort(ranks[:, i])
        ecdf = np.arange(1, N + 1) / N
        color = colors[i % len(colors)]

        y_values = ecdf - sorted_ranks if is_diff else ecdf

        fig.add_trace(
            go.Scatter(
                x=sorted_ranks,
                y=y_values,
                mode="lines",
                name=series_names[i],
                line={"color": color},
                legendgroup=f"group{i}",
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )


def _add_ci_traces(
    fig: go.Figure,
    z_plot: np.ndarray,
    intervals: Dict[str, np.ndarray],
    prob: float,
    row: int = 1,
    col: int = 1,
    is_diff: bool = False,
) -> None:
    """Add confidence interval traces to the plot."""
    # Expected line
    y_expected = [0, 0] if is_diff else [0, 1]
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=y_expected,
            mode="lines",
            name="Expected",
            line={"color": "black", "dash": "dash", "width": 1},
            showlegend=row == 1 and col == 1,
        ),
        row=row,
        col=col,
    )

    # CI traces
    upper = intervals["upper"] - z_plot if is_diff else intervals["upper"]
    lower = intervals["lower"] - z_plot if is_diff else intervals["lower"]

    fig.add_trace(
        go.Scatter(
            x=z_plot,
            y=upper,
            mode="lines",
            name=f"{int(prob*100)}% CI",
            line={"color": "skyblue"},
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=z_plot,
            y=lower,
            mode="lines",
            line={"color": "skyblue"},
            fill="tonexty",
            fillcolor="rgba(135, 206, 235, 0.2)",
            showlegend=row == 1 and col == 1,
            name=f"{int(prob*100)}% CI",
        ),
        row=row,
        col=col,
    )


def plot_ecdf(
    ranks: np.ndarray,
    series_names: List[str],
    prob: float = 0.95,
    K: Optional[int] = None,
    is_diff: bool = False,
) -> go.Figure:
    """
    Plots ECDF with confidence intervals for normalized ranks.

    Args:
        ranks: Array of normalized ranks (NxM, where N is sample size and M is number of series)
        series_names: List of M names for each series
        prob: Desired confidence level (default: 0.95)
        K: Number of evaluation points (default: None, uses min(N,100))
    """
    N = ranks.shape[0]
    if K is None:
        K = min(N, 100)

    z_plot, intervals = _calculate_ci(N, K, prob)

    fig = make_subplots(rows=1, cols=1)
    _add_ecdf_traces(fig, ranks, series_names, is_diff=is_diff)
    _add_ci_traces(fig, z_plot, intervals, prob, is_diff=is_diff)

    title = "Rank ECDF" if not is_diff else "Rank ECDF Difference"
    fig.update_layout(
        title=title,
        xaxis_title="Normalized Rank",
        yaxis_title="ECDF",
        plot_bgcolor="white",
        showlegend=True,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    return fig


def plot_ecdf_combined(
    ranks: np.ndarray,
    series_names: List[str],
    prob: float = 0.95,
    K: Optional[int] = None,
) -> go.Figure:
    """
    Plots ECDF and ECDF difference side by side.

    Args:
        ranks: Array of normalized ranks (NxM, where N is sample size and M is number of series)
        series_names: List of M names for each series
        prob: Desired confidence level (default: 0.95)
        K: Number of evaluation points (default: None, uses min(N,100))
    """
    N = ranks.shape[0]
    if K is None:
        K = min(N, 100)

    z_plot, intervals = _calculate_ci(N, K, prob)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ECDF", "ECDF - Uniform"))

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    _add_ecdf_traces(fig, ranks, series_names, colors, row=1, col=1)
    _add_ecdf_traces(
        fig, ranks, series_names, colors, row=1, col=2, is_diff=True, showlegend=False
    )

    _add_ci_traces(fig, z_plot, intervals, prob, row=1, col=1)
    _add_ci_traces(fig, z_plot, intervals, prob, row=1, col=2, is_diff=True)

    fig.update_layout(
        title="Rank ECDF and Difference",
        showlegend=True,
        plot_bgcolor="white",
        height=400,
    )

    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor="lightgray", title_text="Normalized Rank"
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        title_text="ECDF",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        title_text="ECDF - Uniform",
        row=1,
        col=2,
    )

    return fig
