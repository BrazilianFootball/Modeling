# pylint: disable=too-many-arguments, too-many-positional-arguments

import os
import json

from datetime import datetime as dt
from glob import glob
from time import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from colors import color_mapping

def add_game_result_annotation(fig, date, result, y_offset=0, max_probability=1):
    """Add a game result annotation to the plot with a vertical line and text.

    This function adds a vertical dashed line at the specified date and a text
    annotation showing the game result. The line extends from the bottom of the
    plot to slightly above the maximum probability value, and the text is
    positioned at the top of the line.

    Args:
        fig: The plotly figure object to add the game result annotation to.
        date: The date/time for the game result annotation.
        result (str): The game result text to display.
        y_offset (float, optional): Vertical offset for positioning. Defaults to 0.
        max_champion (float, optional): Maximum champion probability value for line height.
                                        Defaults to 1.

    Returns:
        None: The function modifies the figure object in place.
    """
    fig.add_vline(
        x=date,
        line_dash="dash",
        line_color="gray",
        line_width=2,
        y0=0,
        y1=max_probability+y_offset-.015
    )
    fig.add_annotation(
        x=date,
        y=max_probability+.035+y_offset,
        text=result,
        font={"size": 12, "color": "black"},
        showarrow=False
    )


def add_period(fig, x0, x1, text, text_position, color="gray"):
    """Add a period annotation to the plot with a background rectangle and text.

    This function adds a vertical rectangle (period) to the plot with specified
    start and end dates, along with a text annotation positioned at the given
    coordinates. The rectangle is rendered below other plot elements with
    semi-transparent fill.

    Args:
        fig: The plotly figure object to add the period annotation to.
        x0: Start date/time for the period rectangle.
        x1: End date/time for the period rectangle.
        text (str): Text to display as annotation.
        text_position (tuple): (x, y) coordinates for text placement.
        color (str, optional): Color for the rectangle fill. Defaults to "gray".

    Returns:
        None: The function modifies the figure object in place.
    """
    fig.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor=color,
        opacity=0.3,
        layer="below",
        line_width=0,
    )

    fig.add_annotation(
        x=text_position[0],
        y=text_position[1],
        text=text,
        font={"size": 12, "color": "black"},
        showarrow=False
    )


def add_final_prob(fig, team, results, y_shift=0, col='Champion'):
    """Add final probability annotation to the plot for a specific team.

    This function adds a text annotation showing the final champion probability
    for a given team at the end of the time series. The annotation is positioned
    slightly to the right of the last data point and uses the team's color.

    Args:
        fig: The plotly figure object to add the annotation to.
        team (str): The name of the team to add the final champion probability for.
        results (pd.DataFrame): The results DataFrame containing the team's data.
        y_shift (float, optional): Vertical shift for the annotation. Defaults to 0.
        col (str, optional): The column to plot. Defaults to 'Champion'.

    Returns:
        None: The function modifies the figure object in place.
    """
    prob = results[results['Club'] == team][col].values[-1]
    last_date = results[results['Club'] == team]['Date'].values[-1]
    fig.add_annotation(
        x=pd.to_datetime(last_date)+pd.Timedelta(days=7),
        y=prob+y_shift,
        text=f'{prob:.2%}',
        font={"size": 12, "color": color_mapping[team]},
        showarrow=False
    )


def add_matches_result(fig, results_df, match_result, col):
    """Add match result markers to the plot for a specific result type.

    This function filters the results DataFrame to find matches with a specific
    result (WON, LOST, or DREW) and adds scatter markers to the plot at the
    corresponding dates and champion probabilities. Each result type is
    displayed with a distinct color: green for wins, red for losses, and gray
    for draws. All markers are grouped under the 'Results' legend group.

    Args:
        fig: The plotly figure object to add the match result markers to.
        results_df (pd.DataFrame): DataFrame containing match results with columns
            'Date', 'Champion', and 'Result'.
        match_result (str): The match result type to plot. Must be one of:
            'WON', 'LOST', or 'DREW'.
        col (str): The column to plot.

    Returns:
        None: The function modifies the figure object in place.
    """
    df = results_df[results_df['Match Result'] == match_result]
    colors = {
        'Won': 'green',
        'Lost': 'red',
        'Drew': 'gray',
    }
    fig.add_trace(
        go.Scatter(
            x=df['Match Date'],
            y=df[col],
            mode='markers',
            marker_color=colors[match_result],
            marker_size=8,
            marker_symbol='circle',
            marker_line_width=0,
            name=match_result,
            legendgroup='Results',
        )
    )


def generate_viz(results, club_results, clubs, col, title, subtitle):
    """Generate a visualization plot for team probabilities over time.

    This function creates an interactive line plot showing the probability evolution
    for specified clubs over time. The plot includes:
    - Line traces for each club showing probability progression
    - Period annotations for Club World Cup and FIFA dates
    - Match result markers (won, lost, drew) as colored dots
    - Final probability annotations for each club
    - Grouped legend with clubs and match results

    Args:
        results (pd.DataFrame): DataFrame containing probability data with columns:
            'Club', 'Date', and the column specified by 'col'.
        club_results (pd.DataFrame): DataFrame containing match results with columns:
            'Club', 'Date', 'Match Result', and 'match_date_dt'.
        clubs (list): List of club names to include in the visualization.
        col (str): Column name from 'results' to plot (e.g., 'Champion', 'Z4').
        title (str): Main title for the plot.
        subtitle (str): Subtitle text to display below the main title.

    Returns:
        plotly.graph_objects.Figure: A configured Plotly figure object ready for
            display or export.
    """
    results = results[results['Club'].isin(clubs)]
    results = results[['Club', 'Date', col]]
    results['Date'] = results['Date'] \
        .astype(str) \
        .apply(lambda x: dt.strptime(x, '%d/%m/%Y').strftime('%Y-%m-%d'))

    results = results.merge(club_results, on=['Club', 'Date'], how='left')
    results = results[
        ['Club', 'Date', col, 'Match Result', 'match_date_dt']
    ]
    results.rename(columns={"match_date_dt": "Match Date"}, inplace=True)
    results['Match Date'] = results['Match Date'].astype(str)

    fig = px.line(
        results,
        x='Date',
        y=col,
        color='Club',
        color_discrete_map=color_mapping,
    )

    for trace in fig.data:
        if trace.type == 'scatter' and trace.mode == 'lines':
            trace.legendgroup = 'Clubs'

    add_period(fig, "2025-06-12", "2025-07-12", "Club World Cup", ("2025-06-27", 1.025))
    add_period(fig, "2025-06-02", "2025-06-10", "FIFA Date", ("2025-06-06", 1.025))
    add_period(fig, "2025-09-01", "2025-09-09", "FIFA Date", ("2025-09-05", 1.025))
    add_period(fig, "2025-10-06", "2025-10-14", "FIFA Date", ("2025-10-10", 1.025))
    add_period(fig, "2025-11-10", "2025-11-18", "FIFA Date", ("2025-11-15", 1.025))

    add_matches_result(fig, results, 'Won', col)
    add_matches_result(fig, results, 'Lost', col)
    add_matches_result(fig, results, 'Drew', col)

    for club in clubs:
        add_final_prob(fig, club, results, 0, col)

    fig.update_traces(
        selector={"legendgroup": "Clubs"},
        legendgrouptitle={"text": "Clubs", "font": {"size": 14}},
    )

    fig.update_traces(
        selector={"legendgroup": "Results"},
        legendgrouptitle={"text": "Results", "font": {"size": 14}},
    )

    fig.update_layout(
        title=title+f'<br><span style="font-size: 14px;">{subtitle}</span>',
        xaxis_title='Date',
        yaxis_title='Probability (%)',
        legend_title='Club',
        template="plotly_white",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.01,
            "title_text": None,
        },
        yaxis={
            "tickformat": '.1%',
            "tickmode": "linear",
            "dtick": 0.1,
            "range": [0, 1.05]
        },
        xaxis={
            "title": "Date",
            "type": "date",
            "tickformat": "%d/%m/%Y",
            "tickangle": 45,
            "showgrid": True,
            "gridcolor": "lightgray",
            "showline": True,
            "linecolor": "black"
        },
    )

    return fig


if __name__ == "__main__":
    N_SIMULATIONS = 10_000
    start_time = time()
    year = dt.now().year
    models = {
        'poisson_2': 'Poisson 2',
    }

    with open(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "real_data", "results", "brazil", f"{year}",
            "all_matches.json"
        ),
        "r",
        encoding="utf-8"
    ) as f:
        all_matches = json.load(f)

    simulated_games = pd.DataFrame(all_matches).T.drop(['probabilities'], axis=1).dropna()
    simulated_games['match_date'] = simulated_games['match_datetime'] \
        .apply(
            lambda x: str(x).split()[0].replace("/", "-")
        )

    simulated_games.reset_index(inplace=True)
    simulated_games['index'] = simulated_games['index'].astype(int)
    simulated_games = simulated_games \
        .groupby('match_date') \
        .agg({'index': 'max'}) \
        .reset_index() \
        .sort_values(by='index', ignore_index=True)

    simulated_games['match_date_dt'] = pd.to_datetime(simulated_games['match_date'])
    simulated_games['next_match_date_dt'] = simulated_games['match_date_dt'].shift(-1)
    simulated_games['days_to_next_match'] = (
            simulated_games['next_match_date_dt'] - simulated_games['match_date_dt']
        ).dt.days.fillna(3).astype(int)

    simulated_games['consider'] = False
    skip_count = 0
    for i in simulated_games.index:
        if simulated_games.loc[i]['days_to_next_match'] > 1:
            simulated_games.loc[i, 'consider'] = True
            skip_count = 0
        elif skip_count < 2:
            skip_count += 1
        else:
            simulated_games.loc[i, 'consider'] = True
            skip_count = 0

    simulated_games = simulated_games[simulated_games['consider']]

    games_to_remove = []
    for game in all_matches:
        if "probabilities" in all_matches[game]:
            del all_matches[game]["probabilities"]
        if all_matches[game]["result"] == "TBD":
            games_to_remove.append(game)

    for game in games_to_remove:
        del all_matches[game]

    all_matches = pd.DataFrame(all_matches).T
    all_matches["home_result"] = "Lost"
    all_matches.loc[all_matches["result"] == "H", "home_result"] = "Won"
    all_matches.loc[all_matches["result"] == "D", "home_result"] = "Drew"
    all_matches["away_result"] = "Lost"
    all_matches.loc[all_matches["result"] == "A", "away_result"] = "Won"
    all_matches.loc[all_matches["result"] == "D", "away_result"] = "Drew"
    all_matches["match_date"] = all_matches['match_datetime'] \
        .astype(str) \
        .apply(lambda x: x.split()[0].replace("/", "-"))

    club_results = pd.concat(
        [
            all_matches[["home_team", "match_date", "home_result"]] \
                .rename(columns={"home_team": "Club", "home_result": "Match Result"}),
            all_matches[["away_team", "match_date", "away_result"]] \
                .rename(columns={"away_team": "Club", "away_result": "Match Result"}),
        ],
        ignore_index=True
    )

    club_results = club_results \
        .merge(simulated_games, on='match_date', how='left') \
        .sort_values(by='match_date', ignore_index=True)

    club_results['match_date_dt'] = club_results['match_date_dt'].bfill()
    club_results['match_date'] = club_results['match_date_dt'].astype(str)
    club_results.rename(columns={"match_date": "Date"}, inplace=True)

    for model, model_name in models.items():
        results = pd.DataFrame()
        files = sorted(
            glob(
                os.path.join(os.path.dirname(__file__),
                "..", "..", "real_data", "results", "brazil", f"{year}",
                model, "*", "summary_results.csv")
            )
        )
        for file in files:
            df = pd.read_csv(file)
            results = pd.concat([results, df], ignore_index=True)

        positions = ['Champion', 'G4', 'G5', 'G6', 'G7', 'G8', 'Sula (8-13)', 'Sula (9-14)', 'Z4']
        for position in positions:
            results[position] = (results[position] / N_SIMULATIONS).round(4)

        clubs = ['Flamengo / RJ', 'Palmeiras / SP', 'Cruzeiro / MG']
        title = 'Probability of being champion'
        subtitle = (
            'Probabilities based on 10,000 simulations of the remaining games. '
            'Dots represent the actual results on matches.'
        )
        fig = generate_viz(results, club_results, clubs, 'Champion', title, subtitle)
        y_max = max(results['Champion'])
        add_game_result_annotation(fig, "2025-05-04", "CRU 2x1 FLA", 0, y_max)
        add_game_result_annotation(fig, "2025-05-25", "PAL 0x2 FLA", -0.025, y_max)
        add_game_result_annotation(fig, "2025-06-01", "CRU 2x1 PAL", 0.025, y_max)
        add_game_result_annotation(fig, "2025-10-02", "FLA 0x0 CRU", 0, y_max)
        add_game_result_annotation(fig, "2025-10-19", "FLA 3x2 PAL", -0.025, y_max)
        add_game_result_annotation(fig, "2025-10-26", "PAL 0x0 CRU", 0.025, y_max)

        fig.write_image(
            os.path.join(
                os.path.dirname(__file__), "..", "..",
                "real_data", "results", "brazil", f"{year}",
                f"visualization_{model}_champ.png"
            ),
            width=1600,
            height=800,
        )

        clubs = [
            'Internacional / RS',
            'Vitória / BA',
            'Fortaleza / CE',
            'Ceará / CE',
            'Santos / SP',
        ]
        title = 'Probability of being relegated'
        subtitle = (
            'Probabilities based on 10,000 simulations of the remaining games. '
            'Dots represent the actual results on matches.'
        )
        fig = generate_viz(results, club_results, clubs, 'Z4', title, subtitle)

        fig.write_image(
            os.path.join(
                os.path.dirname(__file__), "..", "..",
                "real_data", "results", "brazil", f"{year}",
                f"visualization_{model}_z4.png"
            ),
            width=1600,
            height=800,
        )

        final_positions_probs_file = sorted(glob(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "real_data", "results", "brazil",
                f"{year}", model, "*", "final_positions_probs.json"
            )
        ))[-1]
        with open(final_positions_probs_file, "r", encoding="utf-8") as f:
            final_positions_probs = json.load(f)

        save_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "real_data", "results", "brazil", f"{year}"
        )

        num_positions = max(len(probs) for probs in final_positions_probs.values())
        positions = [*range(1, num_positions + 1)]

        club_and_max_prob_pos = []
        for club, probs in final_positions_probs.items():
            if probs:
                maxpos = probs.index(max(probs)) if max(probs) != 0 else len(probs) - 1
            else:
                maxpos = num_positions-1
            club_and_max_prob_pos.append((club, maxpos))

        sorted_clubs = [
            club for club, _ in sorted(club_and_max_prob_pos, key=lambda x: x[1], reverse=True)
        ]

        z = []
        for club in sorted_clubs:
            club_probs = final_positions_probs[club]
            row_probs = [
                club_probs[i] if i < len(club_probs) else None for i in range(num_positions)
            ]
            z.append(row_probs)

        z = np.array(z, dtype=float) / N_SIMULATIONS
        z[z == 0] = np.nan
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=positions,
                y=sorted_clubs,
                colorscale="Viridis",
                colorbar={"title": "Probability"},
                zmin=0,
                zmax=1,
                hoverongaps=False
            )
        )
        fig.update_layout(
            title="Final Position Probability for All Clubs",
            xaxis={"title": "Final Position", "tickmode": "linear"},
            yaxis={"title": "Club"},
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.write_image(
            os.path.join(save_dir, 'final_position_probs_all_clubs.png'),
            width=1200,
            height=800,
        )

    print(f"Total time elapsed (visualization): {time() - start_time:,.2f} seconds")
