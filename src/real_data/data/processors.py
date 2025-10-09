from typing import Dict, Any, Tuple
from ..utils.io_utils import save_json, save_csv
from ..utils.path_utils import get_results_path, get_inputs_path

def create_team_mapping(data: Dict[str, Any]) -> Dict[int, str]:
    """Create team index to name mapping."""
    return {
        i + 1: team_name for i, team_name in enumerate(data["team_names"])
    }

def determine_num_teams_by_championship(championship: str) -> int:
    """Determine number of teams based on championship."""
    if championship in ["brazil", "england", "italy", "spain"]:
        return 20
    else:
        return 18

def determine_num_games_by_championship(championship: str) -> int:
    """Determine number of games based on championship."""
    if championship in ["brazil", "england", "italy", "spain"]:
        return 380
    else:
        return 306
