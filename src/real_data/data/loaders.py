from typing import Dict, Any, Tuple
from ..utils.io_utils import load_json, load_csv
from ..utils.path_utils import get_stan_input_file_path, get_results_path

def load_all_matches_data(year: int, championship: str) -> Tuple[Dict[str, Any], str]:
    """Load all matches data for a given year and championship."""
    data_path = os.path.join(
        get_results_path(championship, year), "all_matches.json"
    )
    data = load_json(data_path)
    return data, data_path

def load_stan_input_data(model_name: str, year: int, num_games: int, championship: str) -> Dict[str, Any]:
    """Load Stan input data."""
    model_type = "bradley_terry" if "bradley_terry" in model_name else "poisson"
    file_path = get_stan_input_file_path(championship, year, model_type, num_games)
    return load_json(file_path)

def load_quantiles_data(model_name: str, year: int, num_games: int, championship: str) -> pd.DataFrame:
    """Load quantiles data from CSV."""
    csv_path = os.path.join(
        get_model_results_path(championship, year, model_name, num_games),
        "all_quantiles.csv"
    )
    return load_csv(csv_path)
