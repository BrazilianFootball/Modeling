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
from plots import plot_ecdf, plot_ecdf_combined
from tqdm import tqdm
from utils import load_model_setup

warnings.filterwarnings("ignore")


def calculate_ranks(level: str, model_name: str) -> dict:
    """Calculates normalized ranks for each model parameter."""
    ranks_path = f"results/{level}/{model_name}/ranks.npy"
    if os.path.exists(ranks_path):
        return np.load(ranks_path, allow_pickle=True).item()

    ranks: dict[str, dict[int, list[float]]] = {}
    setup = load_model_setup(level, model_name)

    def _update_ranks(param: str, chain: int, value: float) -> None:
        rank = np.sum(df[param] < value)
        ranks[param] = ranks.get(param, {})
        ranks[param][chain] = ranks[param].get(chain, []) + [rank / len(df)]

    for sim_id in tqdm(setup["data"]):
        path = f"results/{level}/{model_name}/samples/sim_{sim_id}/"

        for chain, file in enumerate(sorted(os.listdir(path))):
            df = pd.read_csv(path + file, comment="#")

            for param, value in setup["data"][sim_id]["variables"].items():
                if isinstance(value, np.ndarray):
                    for i, v in enumerate(value):
                        param_name = f"{param}.{i + 1}"
                        _update_ranks(param_name, chain, v)
                else:
                    _update_ranks(param, chain, value)

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


def generate_plots(
    level: str, model_name: str, ranks: dict, n_sims: int, n_chains: int
) -> None:
    """Generates plots ECDF for all parameters."""
    samples = []
    param_names = []
    points_out_of_bounds = {}
    chain_names = [f"Chain {i}" for i in range(n_chains)]

    save_tasks = []
    plots_dir = f"results/{level}/{model_name}/plots"
    for param in tqdm(ranks.keys(), desc=f"Parameters ({model_name})"):
        sample = np.zeros((n_sims, n_chains))
        for chain in range(n_chains):
            sample[:, chain] = ranks[param][chain]
        samples.append(sample)
        param_names.append(param)
        fig = plot_ecdf_combined(sample, param, chain_names)
        filepath = os.path.join(
            plots_dir, f"{param.replace('.', '_')}_ecdf_combined.png"
        )
        save_tasks.append((fig, filepath))

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
                future.result()

    n_params = len(param_names)
    n_cols = min(4, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, points_out_of_bounds["diff_plot"] = plot_ecdf(
        samples, param_names, chain_names, is_diff=True, n_rows=n_rows, n_cols=n_cols
    )
    for param, points in points_out_of_bounds["diff_plot"].items():
        if points > 0:
            print(f"{param} has {points} points out of bounds on the difference plot")

    if n_rows <= 6:
        fig.write_image(f"results/{level}/{model_name}/plots/all_params_ecdf_diff.png")

    fig, points_out_of_bounds["regular_plot"] = plot_ecdf(
        samples, param_names, chain_names, n_rows=n_rows, n_cols=n_cols
    )
    for param, points in points_out_of_bounds["regular_plot"].items():
        if points > 0:
            print(f"{param} has {points} points out of bounds on the regular plot")

    if n_rows <= 6:
        fig.write_image(f"results/{level}/{model_name}/plots/all_params_ecdf.png")

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
            setup = load_model_setup(level, model_name)
            n_sims = len(setup["data"])
            n_chains = setup["chains"]
            generate_plots(level, model_name, ranks, n_sims, n_chains)


if __name__ == "__main__":
    main()
