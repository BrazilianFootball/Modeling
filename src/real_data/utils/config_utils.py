import logging
from typing import Dict, List

def setup_logging() -> None:
    """Setup logging configuration."""
    cmdstanpy_logger = logging.getLogger("cmdstanpy")
    cmdstanpy_logger.disabled = True

def get_championship_configs() -> Dict[str, Dict]:
    """Get championship configurations."""
    return {
        "20_team": {
            "countries": ["brazil", "england", "italy", "spain"],
            "games": [50, 100, 150, 200, "mid", "end"],
            "total_games": 380
        },
        "18_team": {
            "countries": ["france", "germany", "netherlands", "portugal"],
            "games": [45, 90, 135, 180, "mid", "end"],
            "total_games": 306
        }
    }

def get_model_list() -> List[str]:
    """Get list of models to process."""
    return [
        "bradley_terry_3", "bradley_terry_4",
        "poisson_1", "poisson_2", "poisson_3", "poisson_4", "poisson_5"
    ]

def get_season_list() -> List[int]:
    """Get list of seasons to process."""
    return [*range(2019, 2025)]
