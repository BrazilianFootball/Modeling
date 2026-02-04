# pylint: disable=too-many-locals

import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from constants import MODELS
from plots import _calculate_ci, _check_out_of_bounds, plot_ecdf_combined
from tqdm import tqdm
from utils import load_model_setup

warnings.filterwarnings("ignore")


def calculate_ranks(level: str, model_name: str) -> dict:
    """Calculates normalized ranks for each model parameter."""
    ranks_path = f"results/{level}/{model_name}/ranks.npy"
    if os.path.exists(ranks_path):
        return np.load(ranks_path, allow_pickle=True).item()

    ranks: dict[str, list[float]] = {}
    setup = load_model_setup(level, model_name)

    for sim_id in tqdm(setup["data"]):
        path = f"results/{level}/{model_name}/samples/sim_{sim_id}/"

        dfs = []
        for file in sorted(os.listdir(path)):
            dfs.append(pd.read_csv(path + file, comment="#"))
        df = pd.concat(dfs, ignore_index=True)

        for param, value in setup["data"][sim_id]["variables"].items():
            if isinstance(value, np.ndarray):
                for i, v in enumerate(value):
                    param_name = f"{param}.{i + 1}"
                    rank = np.sum(df[param_name] < v) / len(df)
                    ranks[param_name] = ranks.get(param_name, []) + [rank]
            else:
                rank = np.sum(df[param] < value) / len(df)
                ranks[param] = ranks.get(param, []) + [rank]

    np.save(ranks_path, ranks)
    return ranks


def has_changes(level: str, model_name: str) -> bool:
    """Checks if the model has changed."""
    ranks_path = f"results/{level}/{model_name}/ranks.npy"
    setup_path = f"results/{level}/{model_name}/setup.json"
    plots_dir = f"results/{level}/{model_name}/plots"

    if not os.path.exists(ranks_path) or not os.listdir(plots_dir):
        return True

    ranks_time = os.path.getmtime(ranks_path)
    setup_time = os.path.getmtime(setup_path)
    plots_time = min(os.path.getmtime(p) for p in glob(f"{plots_dir}/*.png"))

    return ranks_time < setup_time or plots_time < ranks_time


def generate_plots(level: str, model_name: str, ranks: dict) -> None:
    """Generates individual ECDF plots for all parameters."""
    samples = []
    param_names = []
    points_out_of_bounds: dict[str, dict[str, int]] = {
        "diff_plot": {},
        "regular_plot": {},
    }
    chain_names = ["Combined"]

    save_tasks = []
    plots_dir = f"results/{level}/{model_name}/plots"
    for param in tqdm(ranks.keys(), desc=f"Parameters ({model_name})"):
        sample = np.array(ranks[param]).reshape(-1, 1)
        samples.append(sample)
        param_names.append(param)

        fig = plot_ecdf_combined(sample, param, chain_names)
        fig.update_layout(showlegend=False)
        filepath = os.path.join(
            plots_dir, f"{param.replace('.', '_')}_ecdf_combined.png"
        )
        save_tasks.append((fig, filepath))

        n = sample.shape[0]
        k = min(n, 100)
        z_plot, intervals = _calculate_ci(n, k, prob=0.95)
        points_out_of_bounds["diff_plot"][param] = _check_out_of_bounds(
            sample, z_plot, intervals, is_diff=True
        )
        points_out_of_bounds["regular_plot"][param] = _check_out_of_bounds(
            sample, z_plot, intervals, is_diff=False
        )

    def save_image(fig_filepath: tuple[go.Figure, str]) -> str:
        fig, filepath = fig_filepath
        fig.write_image(filepath, width=800, height=400, scale=1)
        return filepath

    if save_tasks:
        print(f"Saving {len(save_tasks)} plot images in parallel...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(save_image, task): task for task in save_tasks}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Saving images",
            ):
                try:
                    future.result(timeout=5)
                except TimeoutError:
                    print(f"Timeout error for {future.task_name}")
                    continue

    for param, points in points_out_of_bounds["diff_plot"].items():
        if points > 0:
            print(f"{param} has {points} points out of bounds on the difference plot")

    with open(
        f"results/{level}/{model_name}/points_out_of_bounds.json", "w", encoding="utf-8"
    ) as f:
        json.dump(points_out_of_bounds, f)


def main():
    """Main function for model analysis."""
    for model_name in MODELS:
        print(f"Model: {model_name}")
        level, model_name = model_name.split(".")
        ranks = calculate_ranks(level, model_name)
        if has_changes(level, model_name):
            print(f"Generating plots for {model_name}")
            generate_plots(level, model_name, ranks)


if __name__ == "__main__":
    main()
