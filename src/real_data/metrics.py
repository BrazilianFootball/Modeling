import os

import numpy as np
import pandas as pd
from data_processing import load_all_matches_data


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


def calculate_metrics(model_name: str, year: int, num_rounds: int) -> None:
    """
    Calculate the metrics for a given model and year.

    Args:
        model_name (str): The name of the model to calculate the metrics for.
        year (int): The year of the data to use.
        num_rounds (int): The number of rounds to calculate the metrics for.
    """

    data, _ = load_all_matches_data(year)
    observations = np.zeros(((38 - num_rounds) * 10, 3), dtype=int)
    predictions = np.zeros(((38 - num_rounds) * 10, 3), dtype=float)
    naive_predictions = 1 / 3 * np.ones(((38 - num_rounds) * 10, 3), dtype=float)
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
            .get(str(num_rounds), {})
        ):
            observations[game, :] = results_to_array[game_data["result"]]
            predictions[game, :] = game_data["probabilities"][model_name][
                str(num_rounds)
            ]
            game += 1

    csv_path = "real_data/results/metrics.csv"
    header = [
        "year",
        "model_name",
        "num_rounds",
        "brier_score",
        "ranked_probability_score",
        "log_score",
    ]
    row = [
        year,
        model_name,
        num_rounds,
        brier_score(observations, predictions),
        ranked_probability_score(observations, predictions),
        log_score(observations, predictions),
    ]
    naive_row = [
        year,
        "naive",
        num_rounds,
        brier_score(observations, naive_predictions),
        ranked_probability_score(observations, naive_predictions),
        log_score(observations, naive_predictions),
    ]
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[
            ~(
                (df["year"] == year)
                & (df["model_name"].isin([model_name, "naive"]))
                & (df["num_rounds"] == num_rounds)
            )
        ]
    else:
        df = pd.DataFrame(columns=header)

    df = pd.concat([df, pd.DataFrame([row, naive_row], columns=header)], ignore_index=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
