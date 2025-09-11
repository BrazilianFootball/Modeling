# Based on https://github.com/hyunjimoon/SBC/blob/master/R/plot.R
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals

import numpy as np
import plotly.graph_objects as go
from calculate import adjust_gamma_optimize, ecdf_intervals
from plotly.subplots import make_subplots


def _check_out_of_bounds(
    ranks: np.ndarray,
    z_plot: np.ndarray,
    intervals: dict[str, np.ndarray],
    is_diff: bool = False,
) -> int:
    """Check how many points fall outside the CI in the region 0.005 <= x <= 0.995."""
    n, m = ranks.shape
    out_of_bounds = 0
    for i in range(m):
        sorted_ranks = np.sort(ranks[:, i])
        ecdf = np.arange(1, n + 1) / n
        y_values = ecdf - sorted_ranks if is_diff else ecdf

        y_interp = np.interp(z_plot, sorted_ranks, y_values)

        upper = intervals["upper"] - z_plot if is_diff else intervals["upper"]
        lower = intervals["lower"] - z_plot if is_diff else intervals["lower"]

        mask = (z_plot >= 0.005) & (z_plot <= 0.995)
        out_of_bounds += np.sum(
            (y_interp[mask] > upper[mask]) | (y_interp[mask] < lower[mask])
        )

    return int(out_of_bounds)


def _calculate_ci(
    n: int, k: int, prob: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Calculate confidence intervals for ECDF."""
    gamma = adjust_gamma_optimize(n, k, conf_level=prob)
    z = np.linspace(0, 1, k + 1)
    z_plot = np.concatenate([np.repeat(z[:-1], 2), [1]])

    intervals = ecdf_intervals(n, l=1, k=k, gamma=gamma)
    intervals["upper"] = np.append(intervals["upper"], [1])
    intervals["lower"] = np.append(intervals["lower"], [1])

    return z_plot, intervals


def _add_ecdf_traces(
    fig: go.Figure,
    ranks: np.ndarray,
    series_names: list[str],
    colors: list[str] | None = None,
    row: int = 1,
    col: int = 1,
    is_diff: bool = False,
    showlegend: bool = True,
) -> None:
    """Add ECDF or ECDF difference traces to the plot."""
    n, m = ranks.shape

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

    for i in range(m):
        sorted_ranks = np.sort(ranks[:, i])
        ecdf = np.arange(1, n + 1) / n
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
    intervals: dict[str, np.ndarray],
    prob: float,
    row: int = 1,
    col: int = 1,
    is_diff: bool = False,
) -> None:
    """Add confidence interval traces to the plot."""
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

    upper = intervals["upper"] - z_plot if is_diff else intervals["upper"]
    lower = intervals["lower"] - z_plot if is_diff else intervals["lower"]

    fig.add_trace(
        go.Scatter(
            x=z_plot,
            y=upper,
            mode="lines",
            name=f"{int(prob * 100)}% CI",
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
            name=f"{int(prob * 100)}% CI",
        ),
        row=row,
        col=col,
    )


def plot_ecdf(
    ranks: list[np.ndarray],
    param_names: list[str],
    series_names: list[str],
    prob: float = 0.95,
    k: int | None = None,
    is_diff: bool = False,
    n_rows: int = 1,
    n_cols: int = 1,
    height: int = 1200,
    width: int = 800,
) -> tuple[go.Figure, dict[str, int]]:
    """
    Plots Empirical Cumulative Distribution Functions (ECDFs) with confidence intervals
    for normalized ranks.

    Args:
        ranks: list of normalized rank arrays
               (each array nxm, where n is sample size and m is number of series)
        param_names: list of parameter names for subplot titles
        series_names: list of m names for each series
        prob: Desired confidence level (default: 0.95)
        k: Number of evaluation points (default: None, uses min(n,100))
        is_diff: Whether to plot difference from uniform distribution (default: False)
        n_rows: Number of subplot rows (default: 1)
        n_cols: Number of subplot columns (default: 1)

    Returns:
        Plotly figure containing the ECDF plots, and the number of points out of bounds
    """
    assert len(ranks) == len(param_names), (
        "Number of ranks must match number of parameter names"
    )
    assert len(ranks) <= n_rows * n_cols, (
        "Number of parameters must be less than or equal to the number of subplots"
    )

    subplot_titles = []
    for param in param_names:
        if "log" in param:
            param = param.replace(".", " ").replace("_", " ")
            subplot_titles.append(param)
        else:
            param = param.replace(".", "_{") + "}" if param != "nu" else "nu"
            subplot_titles.append(f"$\\{param}$")

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)

    points_out_of_bounds = {}
    for i, rank_array in enumerate(ranks):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

        n = rank_array.shape[0]
        if k is None:
            k = min(n, 100)

        z_plot, intervals = _calculate_ci(n, k, prob)

        _add_ecdf_traces(
            fig,
            rank_array,
            series_names,
            is_diff=is_diff,
            row=row,
            col=col,
            showlegend=i == 0,
        )
        _add_ci_traces(fig, z_plot, intervals, prob, is_diff=is_diff, row=row, col=col)
        points_out_of_bounds[param_names[i]] = _check_out_of_bounds(
            rank_array, z_plot, intervals, is_diff=is_diff
        )

    fig.update_layout(
        showlegend=True,
        plot_bgcolor="white",
        height=height,
        width=width,
        title_text="Rank ECDF Difference" if is_diff else "Rank ECDF",
    )

    for i in range(1, n_rows * n_cols + 1):
        row = (i - 1) // n_cols + 1
        col = (i - 1) % n_cols + 1
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgray", row=row, col=col
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgray", row=row, col=col
        )

    return fig, points_out_of_bounds


def plot_ecdf_combined(
    ranks: np.ndarray,
    param_name: str,
    series_names: list[str],
    prob: float = 0.95,
    k: int | None = None,
    height: int = 400,
) -> go.Figure:
    """
    Plots ECDF and ECDF difference side by side.

    Args:
        ranks: Array of normalized ranks (nxm, where n is sample size and
               m is number of series)
        param_name: Name of the parameter
        series_names: list of m names for each series
        prob: Desired confidence level (default: 0.95)
        k: Number of evaluation points (default: None, uses min(n,100))
    """
    if "log" in param_name:
        param_name = param_name.replace(".", " ").replace("_", " ")
    else:
        param_name = param_name.replace(".", "_{") + "}" if param_name != "nu" else "nu"

    n = ranks.shape[0]
    if k is None:
        k = min(n, 100)

    z_plot, intervals = _calculate_ci(n, k, prob)

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
        title=(f"$\\{param_name}" + "\\text{ - Rank ECDF and ECDF Difference}$")
        if "log" not in param_name
        else f"{param_name} - Rank ECDF and ECDF Difference",
        showlegend=True,
        plot_bgcolor="white",
        height=height,
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
