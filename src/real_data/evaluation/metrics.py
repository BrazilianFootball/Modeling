# pylint: disable=too-many-locals

import os
import warnings

import numpy as np
import pandas as pd
from typing import Any
from ..data.data_processing import load_all_matches_data
from ..utils.io_utils import load_csv, save_csv

warnings.filterwarnings("ignore")


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


def _safe_log_predictions(predictions: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """Apply safe log transformation to predictions to avoid log(0)."""
    return np.maximum(predictions, epsilon)

def _calculate_cumulative_scores(observations: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate cumulative scores for RPS calculation."""
    n = observations.shape[0]
    total_score = 0.0

    for t in range(n):
        for k in range(1, 3):  # k from 1 to 2
            cumulative_obs = np.sum(observations[t, :k])
            cumulative_pred = np.sum(predictions[t, :k])
            total_score += (cumulative_obs - cumulative_pred) ** 2

    return total_score / (2 * n)

def ranked_probability_score(observations: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate the Ranked Probability Score for probability forecasts."""
    return _calculate_cumulative_scores(observations, predictions)

def log_score(observations: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate the Log Score for probability forecasts."""
    predictions_safe = _safe_log_predictions(predictions)
    return -np.mean(np.sum(observations * np.log(predictions_safe), axis=1))


def _load_quantiles_data(model_name: str, year: int, num_games: int, championship: str) -> pd.DataFrame:
    """Load quantiles data from CSV file."""
    csv_path = os.path.join(
        "..", "real_data", "results", f"{championship}", f"{year}", f"{model_name}",
        f"{str(num_games).zfill(3)}_games", "all_quantiles.csv"
    )
    return load_csv(csv_path)

def _process_quantiles_data(df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Process quantiles data for interval score calculation."""
    df = df[df["team_played"]].reset_index(drop=True)
    df.drop(columns=["team_played", "game_id", "team"], inplace=True)
    real_points = df["real_points"].values
    percentiles = df.columns[:-1]
    return real_points, percentiles, df

def _calculate_interval_penalties(real_points: np.ndarray, lower: np.ndarray,
                                upper: np.ndarray, alpha: float) -> tuple[float, float]:
    """Calculate lower and upper penalties for interval score."""
    lower_penalty = np.mean(2 / alpha * (lower - real_points) * (real_points < lower))
    upper_penalty = np.mean(2 / alpha * (real_points - upper) * (real_points > upper))
    return lower_penalty, upper_penalty

def _calculate_interval_score_for_pair(lower_name: str, upper_name: str, df: pd.DataFrame,
                                     real_points: np.ndarray) -> float:
    """Calculate interval score for a pair of percentiles."""
    interval_range = float(upper_name.replace("p", "")) - float(lower_name.replace("p", ""))
    alpha = round(1 - interval_range / 100, 2)
    lower = df[lower_name].values
    upper = df[upper_name].values

    lower_penalty, upper_penalty = _calculate_interval_penalties(real_points, lower, upper, alpha)
    return lower_penalty + upper_penalty + interval_range

def interval_score(model_name: str, year: int, num_games: int, championship: str) -> float:
    """Calculate the interval score for probabilistic forecasts of real points."""
    if "naive" in model_name:
        return np.inf

    # Load and process data
    df = _load_quantiles_data(model_name, year, num_games, championship)
    real_points, percentiles, df = _process_quantiles_data(df)

    # Calculate score for each percentile pair
    score = 0
    for i, lower_name in enumerate(percentiles):
        upper_name = percentiles[-i-1]
        if lower_name == upper_name:
            break

        score += _calculate_interval_score_for_pair(lower_name, upper_name, df, real_points)

    return score


def _extract_observations_and_predictions(data: dict[str, Any], model_name: str,
                                        num_games: int, sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract observations and predictions from match data."""
    observations = np.zeros((sample_size, 3), dtype=int)
    predictions = np.zeros((sample_size, 3), dtype=float)

    results_to_array = {
        "H": np.array([1, 0, 0]),
        "D": np.array([0, 1, 0]),
        "A": np.array([0, 0, 1]),
    }

    game = 0
    for game_data in data.values():
        if (
            game_data.get("probabilities", {})
            .get(model_name, {})
            .get(str(num_games), {})
        ):
            observations[game, :] = results_to_array[game_data["result"]]
            predictions[game, :] = game_data["probabilities"][model_name][str(num_games)]
            game += 1

    return observations, predictions

def _create_naive_predictions(sample_size: int) -> np.ndarray:
    """Create naive predictions (equal probability for all outcomes)."""
    return 1 / 3 * np.ones((sample_size, 3), dtype=float)

def _calculate_all_metrics(observations: np.ndarray, predictions: np.ndarray,
                          model_name: str, year: int, num_games: int, championship: str) -> tuple[float, float, float, float]:
    """Calculate all metrics for the model."""
    brier = brier_score(observations, predictions)
    rps = ranked_probability_score(observations, predictions)
    log = log_score(observations, predictions)
    interval = interval_score(model_name, year, num_games, championship)

    return brier, rps, log, interval

def _create_metrics_header() -> list[str]:
    """Create header for metrics CSV file."""
    return [
        "year",
        "championship",
        "model_name",
        "num_games",
        "brier_score",
        "ranked_probability_score",
        "log_score",
        "interval_score",
    ]

def _create_model_metrics_row(year: int, championship: str, model_name: str,
                             num_games: int, brier: float, rps: float,
                             log: float, interval: float) -> list:
    """Create metrics row for the model."""
    return [
        year,
        championship,
        model_name,
        num_games,
        brier,
        rps,
        log,
        interval,
    ]

def _create_naive_metrics_row(year: int, championship: str, num_games: int,
                             brier: float, rps: float, log: float) -> list:
    """Create metrics row for naive model."""
    return [
        year,
        championship,
        "naive",
        num_games,
        brier,
        rps,
        log,
        np.nan,
    ]

def _load_existing_metrics(championship: str) -> pd.DataFrame:
    """Load existing metrics CSV file."""
    csv_path = os.path.join("..", "real_data", "results", f"metrics_{championship}.csv")
    if os.path.exists(csv_path):
        return load_csv(csv_path)
    else:
        return pd.DataFrame(columns=_create_metrics_header())

def _remove_duplicate_metrics(df: pd.DataFrame, year: int, championship: str,
                            model_name: str, num_games: int) -> pd.DataFrame:
    """Remove duplicate metrics for the same model/year/games combination."""
    return df[
        ~(
            (df["year"] == year)
            & (df["championship"] == championship)
            & (df["model_name"].isin([model_name, "naive"]))
            & (df["num_games"] == num_games)
        )
    ]

def _save_metrics_to_csv(df: pd.DataFrame, championship: str) -> None:
    """Save metrics DataFrame to CSV file."""
    csv_path = os.path.join("..", "real_data", "results", f"metrics_{championship}.csv")
    save_csv(df, csv_path)

def calculate_metrics(model_name: str, year: int, num_games: int, championship: str) -> None:
    """Calculate the metrics for a given model and year."""
    # Load data
    data, _ = load_all_matches_data(year, championship)
    num_total_matches = len(data)
    sample_size = num_total_matches - num_games

    # Extract observations and predictions
    observations, predictions = _extract_observations_and_predictions(data, model_name, num_games, sample_size)
    naive_predictions = _create_naive_predictions(sample_size)

    # Calculate metrics
    model_brier, model_rps, model_log, model_interval = _calculate_all_metrics(
        observations, predictions, model_name, year, num_games, championship
    )
    naive_brier, naive_rps, naive_log, _ = _calculate_all_metrics(
        observations, naive_predictions, "naive", year, num_games, championship
    )

    # Create rows
    header = _create_metrics_header()
    model_row = _create_model_metrics_row(
        year, championship, model_name, num_games,
        model_brier, model_rps, model_log, model_interval
    )
    naive_row = _create_naive_metrics_row(
        year, championship, num_games,
        naive_brier, naive_rps, naive_log
    )

    # Load existing data and update
    df = _load_existing_metrics(championship)
    df = _remove_duplicate_metrics(df, year, championship, model_name, num_games)
    df = pd.concat([df, pd.DataFrame([model_row, naive_row], columns=header)], ignore_index=True)

    # Save updated metrics
    _save_metrics_to_csv(df, championship)
