# pylint: disable=too-many-arguments, too-many-positional-arguments

import os

from datetime import datetime as dt
from glob import glob

import pandas as pd
import plotly.express as px

from colors import color_mapping

def add_game_result(fig, date, result, y_offset=0, max_probability=1):
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


def add_final_prob(fig, team, results):
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
    prob = results[results['Clube'] == team]['Campeão'].values[-1]
    last_date = results[results['Clube'] == team]['Data'].values[-1]
    fig.add_annotation(
        x=pd.to_datetime(last_date)+pd.Timedelta(days=4),
        y=prob,
        text=f'{prob:.2%}',
        font={"size": 12, "color": color_mapping[team]},
        showarrow=False
    )

if __name__ == "__main__":
    models = {
        'poisson_2': 'Poisson 2',
        'poisson_4': 'Poisson 4',
        'poisson_1': 'Poisson 1',
    }

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
            results['Clube'].isin(['Flamengo / RJ', 'Palmeiras / SP', 'Cruzeiro / MG'])
        ]
        results = results[['Clube', 'Data', 'Campeão']]
        results['Data'] = results['Data'] \
            .astype(str) \
            .apply(lambda x: dt.strptime(x, '%d/%m/%Y').strftime('%Y-%m-%d'))

        fig = px.line(
            results,
            x='Data',
            y='Campeão',
            color='Clube',
            color_discrete_map=color_mapping,
        )

        add_period(fig, "2025-06-12", "2025-07-12", "Copa do Mundo de Clubes", ("2025-06-27", 0.95))
        add_period(fig, "2025-06-02", "2025-06-10", "Data FIFA", ("2025-06-06", 0.95))
        add_period(fig, "2025-09-01", "2025-09-09", "Data FIFA", ("2025-09-05", 0.95))
        add_period(fig, "2025-10-06", "2025-10-14", "Data FIFA", ("2025-10-10", 0.95))
        # add_period(fig, "2025-11-10", "2025-11-18", "Data FIFA", ("2025-11-15", 0.95))

        max_probability = max(results['Campeão'])
        add_game_result(fig, "2025-05-04", "Cruzeiro 2x1 Flamengo", 0, max_probability)
        add_game_result(fig, "2025-05-25", "Palmeiras 0x2 Flamengo", -0.025, max_probability)
        add_game_result(fig, "2025-06-01", "Cruzeiro 2x1 Palmeiras", 0.025, max_probability)
        add_game_result(fig, "2025-10-02", "Flamengo 0x0 Cruzeiro", 0, max_probability)
        add_game_result(fig, "2025-10-19", "Flamengo 3x2 Palmeiras", -0.025, max_probability)
        add_game_result(fig, "2025-10-26", "Palmeiras 0x0 Cruzeiro", 0.025, max_probability)

        add_final_prob(fig, 'Flamengo / RJ', results)
        add_final_prob(fig, 'Palmeiras / SP', results)
        add_final_prob(fig, 'Cruzeiro / MG', results)

        fig.update_layout(
            title=f'{model_name} - Probabilidade de ser campeão',
            xaxis_title='Data',
            yaxis_title='Probabilidade (%)',
            legend_title='Clube',
            template="plotly_white",
            yaxis={
                "tickformat": '.1%',
                "tickmode": "linear",
                "dtick": 0.1,
                "range": [0, 1]
            },
            xaxis={
                "title": "Data",
                "type": "date",
                "tickformat": "%d/%m/%Y",
                "tickangle": 45,
                "showgrid": True,
                "gridcolor": "lightgray",
                "showline": True,
                "linecolor": "black"
            },
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
