# pylint: disable=too-many-locals

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from data_processing import load_all_matches_data

warnings.filterwarnings("ignore")

_METRICS_CACHE: dict[str, list[list[Any]]] = {}

METRICS_HEADER = [
    "year",
    "championship",
    "model_name",
    "num_games",
    "brier_score",
    "ranked_probability_score",
    "log_score",
    "interval_score",
]


def brier_score(observations: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate the Brier Score for probability forecasts.

    The Brier Score measures the accuracy of probabilistic predictions.
    Lower scores indicate better predictions.

    Formula: BS = (1/n) * sum_{t=1 to n} sum_{j=1 to 3} (z_{j,t} - P_{j,t})^2

    Args:
        observations (np.ndarray): Actual outcomes as one-hot encoded vectors (n x 3)
        predictions (np.ndarray): Predicted probabilities (n x 3)

    Returns:
        float: Brier Score
    """
    return np.mean(np.sum((observations - predictions) ** 2, axis=1))


def log_score(observations: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate the Log Score (multinomial log-likelihood) for probability forecasts.

    The Log Score is the only local scoring rule and measures the log-likelihood
    of the observed outcomes given the predicted probabilities.
    Lower scores indicate better predictions.

    Formula: LS = -(1/n) * sum_{t=1 to n} sum_{j=1 to 3} z_{j,t} * log(P_{j,t})

    Args:
        observations (np.ndarray): Actual outcomes as one-hot encoded vectors (n x 3)
        predictions (np.ndarray): Predicted probabilities (n x 3)

    Returns:
        float: Log Score
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    predictions_safe = np.maximum(predictions, epsilon)
    return -np.mean(np.sum(observations * np.log(predictions_safe), axis=1))


def ranked_probability_score(
    observations: np.ndarray, predictions: np.ndarray
) -> float:
    """
    Calculate the Ranked Probability Score for probability forecasts.

    The RPS is the only score that takes into account the ordering of outcomes.
    Lower scores indicate better predictions.

    Formula: RPS = (1/(2n)) * sum_{t=1 to n} sum_{k=1 to 2} (sum_{j=1 to k} (z_{j,t} - P_{j,t}))^2

    Args:
        observations (np.ndarray): Actual outcomes as one-hot encoded vectors (n x 3)
        predictions (np.ndarray): Predicted probabilities (n x 3)

    Returns:
        float: Ranked Probability Score
    """
    n = observations.shape[0]
    total_score = 0.0

    for t in range(n):
        for k in range(1, 3):  # k from 1 to 2
            cumulative_obs = np.sum(observations[t, :k])
            cumulative_pred = np.sum(predictions[t, :k])
            total_score += (cumulative_obs - cumulative_pred) ** 2

    return total_score / (2 * n)


def interval_score(model_name: str, year: int, num_games: int, championship: str) -> float:
    """
    Calculate the interval score for probabilistic forecasts of real points.

    The interval score evaluates the quality of predictive intervals by penalizing intervals
    that are too wide or do not contain the true value. It is computed using quantile predictions
    for each team and compares them to the actual observed points.

    Args:
        model_name (str): The name of the model used to generate the predictions.
        year (int): The year of the competition.
        num_games (int): The number of games considered in the evaluation.
        championship (str): The name of the championship.

    Returns:
        float: The total interval score for all teams in the dataset.
    """
    csv_path = os.path.join(
        "real_data", "results", f"{championship}", f"{year}", f"{model_name}",
        f"{str(num_games).zfill(3)}_games", "all_quantiles.csv"
    )
    df = pd.read_csv(csv_path)
    df = df[df["team_played"]].reset_index(drop=True)
    df.drop(columns=["team_played", "game_id", "team"], inplace=True)
    real_points = df["real_points"].values
    percentiles = df.columns[:-1]
    score = 0
    for i, lower_name in enumerate(percentiles):
        upper_name = percentiles[-i-1]
        if lower_name == upper_name:
            break

        interval_range = float(upper_name.replace("p", "")) - float(lower_name.replace("p", ""))
        alpha = round(1 - interval_range / 100, 2)
        lower = df[lower_name].values
        upper = df[upper_name].values
        lower_penalty = np.mean(2 / alpha * (lower - real_points) * (real_points < lower))
        upper_penalty = np.mean(2 / alpha * (real_points - upper) * (real_points > upper))
        score += lower_penalty + upper_penalty + interval_range

    return score


def calculate_metrics(model_name: str, year: int, num_games: int, championship: str) -> None:
    """
    Calculate the metrics for a given model and year.
    The metrics are stored in cache and written to disk only when flush_metrics_cache() is called.

    Args:
        model_name (str): The name of the model to calculate the metrics for.
        year (int): The year of the data to use.
        num_games (int): The number of games to calculate the metrics for.
        championship (str): The championship of the data.
    """
    data, _ = load_all_matches_data(year, championship)
    num_total_matches = len(data)
    sample_size = num_total_matches - num_games
    observations = np.zeros((sample_size, 3), dtype=int)
    predictions = np.zeros((sample_size, 3), dtype=float)
    game = 0
    results_to_array = {
        "H": np.array([1, 0, 0]),
        "D": np.array([0, 1, 0]),
        "A": np.array([0, 0, 1]),
    }
    for game_data in data.values():
        if (
            game_data.get("probabilities", {})
            .get(model_name, {})
            .get(str(num_games), {})
        ):
            observations[game, :] = results_to_array[game_data["result"]]
            predictions[game, :] = game_data["probabilities"][model_name][
                str(num_games)
            ]
            game += 1

    row = [
        year,
        championship,
        model_name,
        num_games,
        brier_score(observations, predictions),
        ranked_probability_score(observations, predictions),
        log_score(observations, predictions),
        interval_score(model_name, year, num_games, championship),
    ]

    if championship not in _METRICS_CACHE:
        _METRICS_CACHE[championship] = []
    _METRICS_CACHE[championship].append(row)


def flush_metrics_cache() -> None:
    """
    Write all cached metrics to disk and clear the cache.
    """
    for championship, rows in _METRICS_CACHE.items():
        csv_path = os.path.join("real_data", "results", f"metrics_{championship}.csv")

        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
        else:
            df_existing = pd.DataFrame(columns=METRICS_HEADER)

        df_new = pd.DataFrame(rows, columns=METRICS_HEADER)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["year", "championship", "model_name", "num_games"],
            keep="last"
        )

        df_combined.to_csv(csv_path, index=False, encoding="utf-8")

    _METRICS_CACHE.clear()
