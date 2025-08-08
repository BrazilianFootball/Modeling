import json
import os
import shutil
from typing import Any, Callable, Dict

import cmdstanpy
import jsonpickle
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def create_results_dir(model_name: str) -> None:
    """Creates directory to store results."""
    os.makedirs(f"results/{model_name}", exist_ok=True)
    os.makedirs(f"results/{model_name}/plots", exist_ok=True)
    os.makedirs(f"results/{model_name}/samples", exist_ok=True)


def generate_data(generator: Callable, n_sims: int, **kwargs: Any) -> Dict[int, Any]:
    """Generate simulation data using provided generator function.

    Args:
        generator: Function that generates data for a single simulation
        n_sims: Number of simulations to run
        **kwargs: Additional arguments passed to generator

    Returns:
        Dictionary mapping simulation index to generated data
    """
    data: Dict[int, Any] = {}
    new_kwargs = kwargs.copy()
    for i in range(n_sims):
        new_kwargs["seed"] = int(i)
        data[i + 1] = generator(**new_kwargs)

    return data


def save_model_setup(model_name: str, setup: Dict[str, Any]) -> None:
    """Save model setup to JSON file.

    Args:
        model_name: Name of the model
        setup: Dictionary containing model setup
    """
    with open(f"results/{model_name}/setup.json", "w", encoding="utf-8") as file_handle:
        file_handle.write(jsonpickle.encode(setup))


def load_model_setup(model_name: str) -> Dict[str, Any]:
    """Load model setup from JSON file.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary containing model setup
    """
    with open(f"results/{model_name}/setup.json", "r", encoding="utf-8") as file_handle:
        return jsonpickle.decode(file_handle.read())


def create_model_setup(
    model_name: str, data: Dict[int, Dict[str, Any]], **kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Create model setup dictionary.

    Args:
        model_name: Name of the model
        data: Dictionary containing simulation data
        **kwargs: Additional setup parameters

    Returns:
        Dictionary containing complete model setup
    """
    setup = {"model_name": model_name, "data": data}

    setup.update(kwargs)

    return setup


def check_changes(current_setup: str, previous_setup: str) -> bool:
    """Check if there are changes between two setups.

    Args:
        current_setup: JSON string of current setup
        previous_setup: JSON string of previous setup

    Returns:
        True if changes detected, False otherwise
    """
    current_dict = json.loads(current_setup)
    previous_dict = json.loads(previous_setup)

    for key in current_dict:
        if key not in previous_dict or current_dict[key] != previous_dict[key]:
            return True

    for key in previous_dict:
        if key not in current_dict:
            return True

    return False


def setup_was_changed(
    model_name: str, data: Dict[int, Dict[str, Any]], **kwargs: Dict[str, Any]
) -> bool:
    """Check if model setup has changed.

    Args:
        model_name: Name of the model
        data: Dictionary containing simulation data
        **kwargs: Additional setup parameters

    Returns:
        True if setup changed, False otherwise
    """
    current_setup = create_model_setup(model_name, data, **kwargs)
    if not os.path.exists(f"results/{model_name}/setup.json"):
        save_model_setup(model_name, current_setup)
        return True

    previous_setup = load_model_setup(model_name)
    current_json = json.dumps(current_setup, cls=NumpyEncoder, sort_keys=True)
    previous_json = json.dumps(previous_setup, cls=NumpyEncoder, sort_keys=True)
    has_changes = check_changes(current_json, previous_json)
    if has_changes:
        save_model_setup(model_name, current_setup)

    return has_changes


def model_was_changed(model_name: str) -> bool:
    """Check if model has changed.

    Args:
        model_name: Name of the model

    Returns:
        True if model changed, False otherwise
    """
    return os.path.getmtime(f"models/{model_name}.stan") > os.path.getmtime(
        f"results/{model_name}/setup.json"
    )


def clear_dir(dir_path: str) -> None:
    """Remove all files in the directory.

    Args:
        dir_path: Path to the directory
    """
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def remove_model_results(model_name: str) -> None:
    """Remove the model results directory.

    Args:
        model_name: Name of the model
    """
    clear_dir(f"results/{model_name}/samples")
    clear_dir(f"results/{model_name}/plots")
    if os.path.exists(f"results/{model_name}/ranks.npy"):
        os.remove(f"results/{model_name}/ranks.npy")


def run_model(  # pylint: disable=too-many-locals
    model_name: str,
    n_sims: int,
    generator: Callable,
    generator_kwargs: Dict[str, Any],
    model_kwargs: Dict[str, Any],
) -> None:
    """Run model simulations.

    Args:
        model_name: Name of the model
        n_sims: Number of simulations to run
        generator: Function that generates simulation data
        generator_kwargs: Arguments for generator function
        model_kwargs: Arguments for model sampling
    """
    create_results_dir(model_name)
    kwargs = {**generator_kwargs, **model_kwargs}
    data = generate_data(generator, n_sims, **generator_kwargs)
    need_update = setup_was_changed(model_name, data, **kwargs) or model_was_changed(
        model_name
    )
    model = cmdstanpy.CmdStanModel(stan_file=f"models/{model_name}.stan")

    if os.path.exists(f"results/{model_name}/potential_problems.json"):
        with open(
            f"results/{model_name}/potential_problems.json", "r", encoding="utf-8"
        ) as file_handle:
            potential_problems = json.load(file_handle)
    else:
        potential_problems = {}

    if need_update:
        remove_model_results(model_name)

    for i in range(1, n_sims + 1):
        if os.path.exists(f"results/{model_name}/samples/sim_{i}"):
            print(f"Skipping sim {i} of {n_sims} ({model_name})")
            continue

        print(f"Running sim {i} of {n_sims} ({model_name})")
        fit = model.sample(data=data[i]["generated"], **model_kwargs)
        fit.save_csvfiles(f"results/{model_name}/samples/sim_{i}")
        diagnose = fit.diagnose()
        if "no problems detected" not in diagnose:
            potential_problems[i] = diagnose

        os.system("clear")

        if potential_problems:
            with open(
                f"results/{model_name}/potential_problems.json",
                "w",
                encoding="utf-8",
            ) as file_handle:
                json.dump(potential_problems, file_handle)

    if potential_problems:
        print("Potential problems:")
        for i, diagnose in potential_problems.items():
            print(f"Sim {i}:")
            print(diagnose)
            print()

        n_potential_problems = len(potential_problems)
        potential_problems_percentage = n_potential_problems / n_sims
        print(
            f"{n_potential_problems} potential problems detected",
            f"({potential_problems_percentage:.2%} of total)",
        )
