import json
import os
from glob import glob
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


def generate_data(
    generator: Callable[..., Any], n_sims: int, **kwargs: Dict[str, Any]
) -> Dict[int, Any]:
    """Generate simulation data using provided generator function.

    Args:
        generator: Function that generates data for a single simulation
        n_sims: Number of simulations to run
        **kwargs: Additional arguments passed to generator

    Returns:
        Dictionary mapping simulation index to generated data
    """
    data = {}
    for i in range(n_sims):
        kwargs["seed"] = i  # type: ignore
        data[i] = generator(**kwargs)

    return data


def save_model_setup(model_name: str, setup: Dict[str, Any]) -> None:
    """Save model setup to JSON file.

    Args:
        model_name: Name of the model
        setup: Dictionary containing model setup
    """
    with open(
        f"../samples/{model_name}/setup.json", "w", encoding="utf-8"
    ) as file_handle:
        file_handle.write(jsonpickle.encode(setup))


def load_model_setup(model_name: str) -> Dict[str, Any]:
    """Load model setup from JSON file.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary containing model setup
    """
    with open(
        f"../samples/{model_name}/setup.json", "r", encoding="utf-8"
    ) as file_handle:
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
    os.makedirs(f"../samples/{model_name}", exist_ok=True)
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
    if not os.path.exists(f"../samples/{model_name}/setup.json"):
        save_model_setup(model_name, current_setup)
        return True

    previous_setup = load_model_setup(model_name)
    current_json = json.dumps(current_setup, cls=NumpyEncoder, sort_keys=True)
    previous_json = json.dumps(previous_setup, cls=NumpyEncoder, sort_keys=True)
    has_changes = check_changes(current_json, previous_json)
    if has_changes:
        save_model_setup(model_name, current_setup)

    return has_changes


def remove_empty_dirs(model_name: str) -> None:
    """Remove empty directories in model samples folder.

    Args:
        model_name: Name of the model
    """
    for root, directories, _ in os.walk(f"../samples/{model_name}", topdown=False):
        for directory in directories:
            dir_path = os.path.join(root, directory)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)


def clear_samples(model_name: str) -> None:
    """Clear all sample files for a model.

    Args:
        model_name: Name of the model
    """
    for file in glob(f"../samples/{model_name}/**/*"):
        if file.endswith(".csv"):
            os.remove(file)

    for file in glob(f"../samples/{model_name}/potential_problems.json"):
        os.remove(file)

    remove_empty_dirs(model_name)


# pylint: disable=too-many-locals
def run_model(
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
    kwargs = {**generator_kwargs, **model_kwargs}
    data = generate_data(generator, n_sims, **generator_kwargs)
    need_update = setup_was_changed(model_name, data, **kwargs)

    if need_update:
        clear_samples(model_name)
        potential_problems = {}
        model = cmdstanpy.CmdStanModel(stan_file=f"../models/{model_name}.stan")
        for i in range(n_sims):
            print(f"Running sim {i+1} of {n_sims} ({model_name})")
            fit = model.sample(data=data[i]["generated"], **model_kwargs)
            fit.save_csvfiles(f"../samples/{model_name}/sim_{i+1}")
            diagnose = fit.diagnose()
            if "no problems detected" not in diagnose:
                potential_problems[i + 1] = diagnose

            os.system("clear")

        if potential_problems:
            with open(
                f"../samples/{model_name}/potential_problems.json",
                "w",
                encoding="utf-8",
            ) as file_handle:
                json.dump(potential_problems, file_handle)

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
