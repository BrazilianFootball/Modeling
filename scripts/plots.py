# Based on https://github.com/hyunjimoon/SBC/blob/master/R/plot.R

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
from scipy.stats import binom

from scripts.calculate import adjust_gamma_optimize, ecdf_intervals


def plot_rank_hist(
    ranks: np.ndarray,
    series_names: List[str],
    bins: Optional[int] = None,
    prob: float = 0.95,
) -> go.Figure:
    """
    Plots histogram of ranks with confidence interval.

    Args:
        ranks: Array of normalized ranks (NxM, where N is sample size and M is number of series)
        series_names: List of M names for each series
        bins: Number of bins (default: Sturges rule)
        prob: Confidence level (default: 0.95)
    """
    N, M = ranks.shape

    if bins is None:
        bins = int(np.ceil(np.log2(N) + 1))

    expected = 1.0
    alpha = 1 - prob
    ci_lower = binom.ppf(alpha / 2, N, 1 / bins) / (N * (1.0 / bins))
    ci_upper = binom.ppf(1 - alpha / 2, N, 1 / bins) / (N * (1.0 / bins))

    fig = go.Figure()

    # Histograms
    for i in range(M):
        fig.add_trace(
            go.Histogram(
                x=ranks[:, i],
                nbinsx=bins,
                name=series_names[i],
                opacity=0.7,
                histnorm="probability density",
                xbins=dict(start=0, end=1, size=(1.0 / bins)),
            )
        )

    # Expected line and intervals
    x_range = [0, 1]
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=[expected, expected],
            mode="lines",
            name="Expected",
            line=dict(color="black", dash="dash", width=1),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=[ci_upper, ci_upper],
            mode="lines",
            name=f"{int(prob*100)}% CI",
            line=dict(color="skyblue", dash="dot"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=[ci_lower, ci_lower],
            mode="lines",
            line=dict(color="skyblue", dash="dot"),
            showlegend=False,
            fill="tonexty",
            fillcolor="rgba(135, 206, 235, 0.2)",
        )
    )

    fig.update_layout(
        title="Rank Density",
        xaxis_title="Normalized Rank",
        yaxis_title="Density",
        plot_bgcolor="white",
        showlegend=True,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    return fig


def plot_ecdf(
    ranks: np.ndarray,
    series_names: List[str],
    prob: float = 0.95,
    K: Optional[int] = None,
) -> go.Figure:
    """
    Plots ECDF with confidence intervals for normalized ranks.

    Args:
        ranks: Array of normalized ranks (NxM, where N is sample size and M is number of series)
        series_names: List of M names for each series
        prob: Desired confidence level (default: 0.95)
        K: Number of evaluation points (default: None, uses min(N,100))
    """
    N, M = ranks.shape
    if K is None:
        K = min(N, 100)

    gamma = adjust_gamma_optimize(N, K, conf_level=prob)

    z = np.linspace(0, 1, K + 1)
    z_plot = np.concatenate([np.repeat(z[:-1], 2), [1]])

    # Calculate intervals
    intervals = ecdf_intervals(N, L=1, K=K, gamma=gamma)
    intervals["upper"] = np.append(intervals["upper"], [1])
    intervals["lower"] = np.append(intervals["lower"], [1])

    fig = go.Figure()

    # ECDF for each series
    for i in range(M):
        sorted_ranks = np.sort(ranks[:, i])
        ecdf = np.arange(1, N + 1) / N

        fig.add_trace(
            go.Scatter(x=sorted_ranks, y=ecdf, mode="lines", name=series_names[i])
        )

    # Expected line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Expected",
            line=dict(color="black", dash="dash", width=1),
        )
    )

    # Confidence intervals
    fig.add_trace(
        go.Scatter(
            x=z_plot,
            y=intervals["upper"],
            mode="lines",
            name=f"{int(prob*100)}% CI",
            line=dict(color="skyblue"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=z_plot,
            y=intervals["lower"],
            mode="lines",
            line=dict(color="skyblue"),
            fill="tonexty",
            fillcolor="rgba(135, 206, 235, 0.2)",
            showlegend=True,
            name=f"{int(prob*100)}% CI",
        )
    )

    fig.update_layout(
        title="Rank ECDF",
        xaxis_title="Normalized Rank",
        yaxis_title="ECDF",
        plot_bgcolor="white",
        showlegend=True,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    return fig


def plot_ecdf_diff(
    ranks: np.ndarray,
    series_names: List[str],
    prob: float = 0.95,
    K: Optional[int] = None,
) -> go.Figure:
    """
    Plots ECDF difference from uniform.

    Args:
        ranks: Array of normalized ranks (NxM, where N is sample size and M is number of series)
        series_names: List of M names for each series
        prob: Confidence level (default: 0.95)
        K: Number of evaluation points (default: None)
    """
    N, M = ranks.shape
    if K is None:
        K = min(N, 100)

    gamma = adjust_gamma_optimize(N, K, conf_level=prob)

    z = np.linspace(0, 1, K + 1)
    z_plot = np.concatenate([np.repeat(z[:-1], 2), [1]])

    # Calculate intervals
    intervals = ecdf_intervals(N, L=1, K=K, gamma=gamma)
    intervals["upper"] = np.append(intervals["upper"], [1])
    intervals["lower"] = np.append(intervals["lower"], [1])

    fig = go.Figure()

    # ECDF difference for each series
    for i in range(M):
        sorted_ranks = np.sort(ranks[:, i])
        ecdf = np.arange(1, N + 1) / N

        fig.add_trace(
            go.Scatter(
                x=sorted_ranks,
                y=ecdf - sorted_ranks,
                mode="lines",
                name=series_names[i],
            )
        )

    # Expected line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 0],
            mode="lines",
            name="Expected",
            line=dict(color="black", dash="dash", width=1),
        )
    )

    # Confidence intervals
    fig.add_trace(
        go.Scatter(
            x=z_plot,
            y=intervals["upper"] - z_plot,
            mode="lines",
            name=f"{int(prob*100)}% CI",
            line=dict(color="skyblue"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=z_plot,
            y=intervals["lower"] - z_plot,
            mode="lines",
            line=dict(color="skyblue"),
            fill="tonexty",
            fillcolor="rgba(135, 206, 235, 0.2)",
            showlegend=True,
            name=f"{int(prob*100)}% CI",
        )
    )

    fig.update_layout(
        title="Rank ECDF Difference",
        xaxis_title="Normalized Rank",
        yaxis_title="ECDF - Uniform",
        plot_bgcolor="white",
        showlegend=True,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    return fig
