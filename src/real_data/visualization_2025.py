# pylint: disable=too-many-arguments, too-many-positional-arguments

import os
import json

from datetime import datetime as dt
from glob import glob

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
        max_probability (float, optional): Maximum probability value for line height. Defaults to 1.

    Returns:
        None: The function modifies the figure object in place.
    """
    fig.add_vline(
        x=date,
        line_dash="dash",
        line_color="gray",
        line_width=2,
        y0=0,
        y1=max_probability+.025+y_offset
    )
    fig.add_annotation(
        x=date,
        y=max_probability+.025+y_offset,
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


def add_final_prob(fig, team, results, y_shift=0):
    """Add final probability annotation to the plot for a specific team.

    This function adds a text annotation showing the final championship probability
    for a given team at the end of the time series. The annotation is positioned
    slightly to the right of the last data point and uses the team's color.

    Args:
        fig: The plotly figure object to add the annotation to.
        team (str): The name of the team to add the final probability for.
        results (pd.DataFrame): The results DataFrame containing the team's data.

    Returns:
        None: The function modifies the figure object in place.
    """
    prob = results[results['Club'] == team]['Championship Probability'].values[-1]
    last_date = results[results['Club'] == team]['Date'].values[-1]
    fig.add_annotation(
        x=pd.to_datetime(last_date)+pd.Timedelta(days=7),
        y=prob+y_shift,
        text=f'{prob:.2%}',
        font={"size": 12, "color": color_mapping[team]},
        showarrow=False
    )

def add_matches_result(fig, results_df, match_result):
    """Add match result markers to the plot for a specific result type.

    This function filters the results DataFrame to find matches with a specific
    result (WON, LOST, or DREW) and adds scatter markers to the plot at the
    corresponding dates and championship probabilities. Each result type is
    displayed with a distinct color: green for wins, red for losses, and gray
    for draws. All markers are grouped under the 'Results' legend group.

    Args:
        fig: The plotly figure object to add the match result markers to.
        results_df (pd.DataFrame): DataFrame containing match results with columns
            'Date', 'Championship Probability', and 'Result'.
        match_result (str): The match result type to plot. Must be one of:
            'WON', 'LOST', or 'DREW'.

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
            y=df['Championship Probability'],
            mode='markers',
            marker_color=colors[match_result],
            marker_size=8,
            marker_symbol='circle',
            marker_line_width=0,
            name=match_result,
            legendgroup='Results',
        )
    )

if __name__ == "__main__":
    models = {
        'poisson_2': 'Poisson 2',
        'poisson_4': 'Poisson 4',
        'poisson_1': 'Poisson 1',
    }

    with open(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "real_data", "results", "brazil", "2025",
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
                "..", "..", "real_data", "results", "brazil", "2025",
                model, "*", "summary_results.csv")
            )
        )
        for file in files:
            df = pd.read_csv(file)
            results = pd.concat([results, df], ignore_index=True)

        results = results[
            results['Club'].isin(['Flamengo / RJ', 'Palmeiras / SP', 'Cruzeiro / MG'])
        ]
        results = results[['Club', 'Date', 'Championship Probability']]
        results['Date'] = results['Date'] \
            .astype(str) \
            .apply(lambda x: dt.strptime(x, '%d/%m/%Y').strftime('%Y-%m-%d'))

        results = results.merge(club_results, on=['Club', 'Date'], how='left')
        results = results[
            ['Club', 'Date', 'Championship Probability', 'Match Result', 'match_date_dt']
        ]
        results.rename(columns={"match_date_dt": "Match Date"}, inplace=True)
        results['Match Date'] = results['Match Date'].astype(str)

        fig = px.line(
            results,
            x='Date',
            y='Championship Probability',
            color='Club',
            color_discrete_map=color_mapping,
        )

        for trace in fig.data:
            if trace.type == 'scatter' and trace.mode == 'lines':
                trace.legendgroup = 'Clubs'

        add_period(fig, "2025-06-12", "2025-07-12", "Club World Cup", ("2025-06-27", 0.95))
        add_period(fig, "2025-06-02", "2025-06-10", "FIFA Date", ("2025-06-06", 0.95))
        add_period(fig, "2025-09-01", "2025-09-09", "FIFA Date", ("2025-09-05", 0.95))
        add_period(fig, "2025-10-06", "2025-10-14", "FIFA Date", ("2025-10-10", 0.95))
        # add_period(fig, "2025-11-10", "2025-11-18", "FIFA Date", ("2025-11-15", 0.95))

        y_max = max(results['Championship Probability'])
        add_game_result_annotation(fig, "2025-05-04", "Cruzeiro 2x1 Flamengo", 0, y_max)
        add_game_result_annotation(fig, "2025-05-25", "Palmeiras 0x2 Flamengo", -0.025, y_max)
        add_game_result_annotation(fig, "2025-06-01", "Cruzeiro 2x1 Palmeiras", 0.025, y_max)
        add_game_result_annotation(fig, "2025-10-02", "Flamengo 0x0 Cruzeiro", 0, y_max)
        add_game_result_annotation(fig, "2025-10-19", "Flamengo 3x2 Palmeiras", -0.025, y_max)
        add_game_result_annotation(fig, "2025-10-26", "Palmeiras 0x0 Cruzeiro", 0.025, y_max)

        add_matches_result(fig, results, 'Won')
        add_matches_result(fig, results, 'Lost')
        add_matches_result(fig, results, 'Drew')

        add_final_prob(fig, 'Flamengo / RJ', results, -0.0125)
        add_final_prob(fig, 'Palmeiras / SP', results, 0.0125)
        add_final_prob(fig, 'Cruzeiro / MG', results, 0)

        title = 'Probability of being champion'
        subtitle = (
            'Probabilities based on 100,000 simulations of the remaining games. '
            'Dots represent the actual results on matches.'
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
                "range": [0, 1]
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

        fig.update_traces(
            selector={"legendgroup": "Clubs"},
            legendgrouptitle={"text": "Clubs", "font": {"size": 14}},
        )

        fig.update_traces(
            selector={"legendgroup": "Results"},
            legendgrouptitle={"text": "Results", "font": {"size": 14}},
        )

        fig.write_image(
            os.path.join(
                os.path.dirname(__file__), "..", "..",
                "real_data", "results", "brazil", "2025",
                f"visualization_{model}.png"
            ),
            width=1600,
            height=800,
        )
