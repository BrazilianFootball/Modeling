import os
from typing import Dict, List

import numpy as np
import pandas as pd
from plots import plot_ecdf
from tqdm import tqdm
from utils import load_model_setup

MODELS = [
    "bt_model_1",
    "bt_model_2",
    "poisson_model_1",
    "poisson_model_2",
]


def create_results_dirs(models: List[str]) -> None:
    """Creates directories to store results."""
    for model_name in models:
        os.makedirs(f"../results/{model_name}", exist_ok=True)


def calculate_ranks(model_name: str) -> Dict:
    """Calculates normalized ranks for each model parameter."""
    ranks: Dict[str, Dict[int, List[float]]] = {}
    setup = load_model_setup(model_name)

    def _update_ranks(param: str, chain: int, max_rank: float) -> None:
        rank = df[param].rank(ascending=False)[0]
        ranks[param] = ranks.get(param, {})
        ranks[param][chain] = ranks[param].get(chain, []) + [rank / max_rank]

    for sim_id in tqdm(setup["data"]):
        path = f"../samples/{model_name}/sim_{sim_id}/"

        for chain, file in enumerate(sorted(os.listdir(path))):
            df = pd.read_csv(path + file, comment="#")
            max_rank = len(df) + 1

            for param, value in setup["data"][sim_id]["variables"].items():
                if isinstance(value, np.ndarray):
                    for idx in range(len(value)):
                        param_name = f"{param}.{idx+1}"
                        _update_ranks(param_name, chain, max_rank)
                else:
                    _update_ranks(param, chain, max_rank)

    return ranks


def generate_plots(model_name: str, ranks: Dict, n_sims: int, n_chains: int) -> None:
    """Generates ECDF plots for each parameter."""
    for param in tqdm(ranks.keys(), desc="Parameters"):
        sample = np.zeros((n_sims, n_chains))
        for chain in range(n_chains):
            sample[:, chain] = ranks[param][chain]

        chain_names = [f"chain_{i}" for i in range(n_chains)]

        # Plot ECDF difference
        fig = plot_ecdf([sample], [param], chain_names, is_diff=True)
        fig.write_image(f"../results/{model_name}/{param}_ecdf_diff.png")

        # Plot ECDF
        fig = plot_ecdf([sample], [param], chain_names)
        fig.write_image(f"../results/{model_name}/{param}_ecdf.png")


def main():
    """Main function for model analysis."""
    create_results_dirs(MODELS)
    for model_name in MODELS:
        ranks = calculate_ranks(model_name)
        setup = load_model_setup(model_name)
        n_sims = len(setup["data"])
        n_chains = len(os.listdir(f"../samples/{model_name}/sim_1/"))
        generate_plots(model_name, ranks, n_sims, n_chains)


if __name__ == "__main__":
    main()
