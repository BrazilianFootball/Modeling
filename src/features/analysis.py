import os
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from plots import plot_ecdf, plot_ecdf_combined
from tqdm import tqdm
from utils import load_model_setup

warnings.filterwarnings("ignore")

MODELS = [
    "bradley_terry_1",
    "bradley_terry_2",
    "bradley_terry_3",
    "bradley_terry_4",
    "poisson_model_1",
    "poisson_model_2",
    # "bad_prior_example",
    # "nice_prior_example",
]


def calculate_ranks(model_name: str) -> Dict:
    """Calculates normalized ranks for each model parameter."""
    ranks_path = f"results/{model_name}/ranks.npy"
    if os.path.exists(ranks_path):
        return np.load(ranks_path, allow_pickle=True).item()

    ranks: Dict[str, Dict[int, List[float]]] = {}
    setup = load_model_setup(model_name)

    def _update_ranks(param: str, chain: int, value: float) -> None:
        rank = np.sum(df[param] < value)
        ranks[param] = ranks.get(param, {})
        ranks[param][chain] = ranks[param].get(chain, []) + [rank / len(df)]

    for sim_id in tqdm(setup["data"]):
        path = f"results/{model_name}/samples/sim_{sim_id}/"

        for chain, file in enumerate(sorted(os.listdir(path))):
            df = pd.read_csv(path + file, comment="#")

            for param, value in setup["data"][sim_id]["variables"].items():
                if isinstance(value, np.ndarray):
                    for i, v in enumerate(value):
                        param_name = f"{param}.{i+1}"
                        _update_ranks(param_name, chain, v)
                else:
                    _update_ranks(param, chain, value)

    np.save(ranks_path, ranks)
    return ranks


def has_changes(model_name: str) -> bool:
    """Checks if the model has changed."""
    ranks_path = f"results/{model_name}/ranks.npy"
    setup_path = f"results/{model_name}/setup.json"
    plots_dir = f"results/{model_name}/plots"

    if not os.path.exists(ranks_path) or not os.listdir(plots_dir):
        return True

    ranks_time = os.path.getmtime(ranks_path)
    setup_time = os.path.getmtime(setup_path)
    plots_time = os.path.getmtime(f"{plots_dir}/all_params_ecdf.png")

    return ranks_time < setup_time or plots_time < ranks_time


def generate_plots(model_name: str, ranks: Dict, n_sims: int, n_chains: int) -> None:
    """Generates plots ECDF for all parameters."""
    samples = []
    param_names = []
    chain_names = [f"chain_{i}" for i in range(n_chains)]

    for param in tqdm(ranks.keys(), desc=f"Parameters ({model_name})"):
        sample = np.zeros((n_sims, n_chains))
        for chain in range(n_chains):
            sample[:, chain] = ranks[param][chain]
        samples.append(sample)
        param_names.append(param)
        fig = plot_ecdf_combined(sample, param, chain_names)
        fig.write_image(f"results/{model_name}/plots/{param}_ecdf_combined.png")

    n_params = len(param_names)
    n_cols = min(4, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig = plot_ecdf(
        samples, param_names, chain_names, is_diff=True, n_rows=n_rows, n_cols=n_cols
    )
    fig.write_image(f"results/{model_name}/plots/all_params_ecdf_diff.png")

    fig = plot_ecdf(samples, param_names, chain_names, n_rows=n_rows, n_cols=n_cols)
    fig.write_image(f"results/{model_name}/plots/all_params_ecdf.png")


def main():
    """Main function for model analysis."""
    for model_name in MODELS:
        ranks = calculate_ranks(model_name)
        if has_changes(model_name):
            setup = load_model_setup(model_name)
            n_sims = len(setup["data"])
            n_chains = setup["chains"]
            generate_plots(model_name, ranks, n_sims, n_chains)


if __name__ == "__main__":
    main()
