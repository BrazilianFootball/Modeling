# pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-instance-attributes
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import cmdstanpy
import pandas as pd


@dataclass
class PlayerStats:
    """Statistics for a player across seasons."""

    name: str = ""
    minutes_per_team: dict[str, int] = field(default_factory=dict)
    total_minutes: int = 0
    total_games: int = 0
    total_wins: int = 0
    total_draws: int = 0
    weighted_score: float = 0.0
    weighted_goals: float = 0.0
    weighted_goals_difference: float = 0.0

    @property
    def main_team(self) -> str:
        """Returns the team where the player had the most minutes."""
        if not self.minutes_per_team:
            return ""
        return max(self.minutes_per_team, key=lambda k: self.minutes_per_team[k])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "team": self.main_team,
            "total_minutes": self.total_minutes,
            "total_games": self.total_games,
            "total_wins": self.total_wins,
            "total_draws": self.total_draws,
            "weighted_score": self.weighted_score,
            "weighted_goals": self.weighted_goals,
            "weighted_goals_difference": self.weighted_goals_difference,
        }


def _get_data_path() -> str:
    """
    Returns the path to the processed data directory.

    Returns:
        Path to the processed data directory.
    """
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "Data", "results", "processed"
    )


def _load_season_squads_data(season: int) -> dict[str, Any]:
    """
    Loads the squads data for a specific season.

    Args:
        season: Year of the season to be loaded.

    Returns:
        Dictionary with the squads data for the season.
    """
    data_path = _get_data_path()
    file_name = f"Serie_A_{season}_squads.json"
    file_path = os.path.join(data_path, file_name)

    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def _load_season_games_data(season: int) -> dict[str, Any]:
    """
    Loads the games data for a specific season.

    Args:
        season: Year of the season to be loaded.

    Returns:
        Dictionary with the games data for the season.
    """
    data_path = _get_data_path()
    file_name = f"Serie_A_{season}_games.json"
    file_path = os.path.join(data_path, file_name)

    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def _build_player_info_lookup(seasons: list[int]) -> dict[str, tuple[str, str]]:
    """
    Builds a lookup dictionary mapping player IDs to (name, team) tuples.

    Args:
        seasons: List of years of the seasons to process.

    Returns:
        Dictionary mapping player ID to (player_name, team_name) tuple.
    """
    player_info: dict[str, tuple[str, str]] = {}

    for season in seasons:
        games_data = _load_season_games_data(season)
        for game in games_data.values():
            for player_data in game["Players"]:
                player_id = player_data[0][-6:]
                if player_id not in player_info:
                    player_info[player_id] = (player_data[0], player_data[1])

    return player_info


def _process_seasons_data(
    seasons: list[int],
) -> tuple[dict[str, Any], dict[str, int], dict[str, PlayerStats]]:
    """
    Processes the data from multiple seasons and generates the input data for the Stan model.

    Args:
        seasons: List of years of the seasons to be processed.

    Returns:
        Tuple containing:
            - stan_data: Dictionary with the formatted data for the Stan model.
            - players_mapping: Dictionary mapping player identifier to index.
            - players_stats: Dictionary with detailed statistics for each player.
    """
    players_mapping: dict[str, int] = {"None": 1}
    players_stats: dict[str, PlayerStats] = {}
    stan_data: dict[str, Any] = {
        "home_players": [],
        "home_players_minutes": [],
        "away_players": [],
        "away_players_minutes": [],
        "home_goals": [],
        "away_goals": [],
    }

    player_info_lookup = _build_player_info_lookup(seasons)

    for season in seasons:
        squads_data = _load_season_squads_data(season)
        games_data = _load_season_games_data(season)

        game_teams_lookup: dict[str, tuple[str, str]] = {}
        for game_id, game in games_data.items():
            game_teams_lookup[game_id] = (game["Home"], game["Away"])

        for game_id, game_data in squads_data.items():
            home_goals, away_goals = list(
                map(int, game_data["Summary"]["Result"].upper().split(" X "))
            )

            if home_goals > away_goals:
                home_result = "win"
                away_result = "loss"
            elif home_goals < away_goals:
                home_result = "loss"
                away_result = "win"
            else:
                home_result = "draw"
                away_result = "draw"

            home_team, away_team = game_teams_lookup.get(game_id, ("", ""))

            del game_data["Summary"]

            home_players: dict[str, int] = {}
            away_players: dict[str, int] = {}

            home_players_in_game: set[str] = set()
            away_players_in_game: set[str] = set()

            for sub_game_data in game_data.values():
                if sub_game_data["Time"] == 0:
                    continue

                sub_game_time = sub_game_data["Time"]

                for player in sub_game_data["Home"]["Squad"]:
                    players_mapping[player] = players_mapping.setdefault(
                        player, len(players_mapping) + 1
                    )

                    if player not in players_stats:
                        player_name, _ = player_info_lookup.get(
                            player, ("", "")
                        )
                        players_stats[player] = PlayerStats(name=player_name)

                    players_stats[player].total_minutes += sub_game_time
                    if home_team:
                        players_stats[player].minutes_per_team[home_team] = (
                            players_stats[player].minutes_per_team.get(home_team, 0)
                            + sub_game_time
                        )

                    home_players[player] = (
                        home_players.get(player, 0) + sub_game_data["Time"]
                    )
                    home_players_in_game.add(player)

                for player in sub_game_data["Away"]["Squad"]:
                    players_mapping[player] = players_mapping.setdefault(
                        player, len(players_mapping) + 1
                    )

                    if player not in players_stats:
                        player_name, _ = player_info_lookup.get(
                            player, ("", "")
                        )
                        players_stats[player] = PlayerStats(name=player_name)

                    players_stats[player].total_minutes += sub_game_time
                    if away_team:
                        players_stats[player].minutes_per_team[away_team] = (
                            players_stats[player].minutes_per_team.get(away_team, 0)
                            + sub_game_time
                        )

                    away_players[player] = (
                        away_players.get(player, 0) + sub_game_data["Time"]
                    )
                    away_players_in_game.add(player)

            for player in home_players_in_game:
                players_stats[player].total_games += 1
                minutes_in_game = home_players[player]
                weight = minutes_in_game / 90

                if home_result == "win":
                    players_stats[player].total_wins += 1
                    players_stats[player].weighted_score += weight * 3
                elif home_result == "draw":
                    players_stats[player].total_draws += 1
                    players_stats[player].weighted_score += weight * 1

                players_stats[player].weighted_goals += weight * home_goals
                players_stats[player].weighted_goals_difference += (
                    weight * (home_goals - away_goals)
                )

            for player in away_players_in_game:
                players_stats[player].total_games += 1
                minutes_in_game = away_players[player]
                weight = minutes_in_game / 90

                if away_result == "win":
                    players_stats[player].total_wins += 1
                    players_stats[player].weighted_score += weight * 3
                elif away_result == "draw":
                    players_stats[player].total_draws += 1
                    players_stats[player].weighted_score += weight * 1

                players_stats[player].weighted_goals += weight * away_goals
                players_stats[player].weighted_goals_difference += (
                    weight * (away_goals - home_goals)
                )

            if sum(home_players.values()) < 990:
                home_players["None"] = 990 - sum(home_players.values())
            if sum(away_players.values()) < 990:
                away_players["None"] = 990 - sum(away_players.values())

            stan_data["home_players"].append(
                [players_mapping[x] for x in home_players]
            )
            stan_data["home_players_minutes"].append(list(home_players.values()))
            stan_data["away_players"].append(
                [players_mapping[x] for x in away_players]
            )
            stan_data["away_players_minutes"].append(list(away_players.values()))
            stan_data["home_goals"].append(home_goals)
            stan_data["away_goals"].append(away_goals)

    num_players_per_game = max(
        [len(x) for x in stan_data["home_players"]]
        + [len(x) for x in stan_data["away_players"]]
    )

    stan_data["num_games"] = len(stan_data["home_goals"])
    stan_data["num_players"] = len(players_mapping)
    stan_data["num_players_per_game"] = num_players_per_game

    for i in range(stan_data["num_games"]):
        while len(stan_data["home_players"][i]) < num_players_per_game:
            stan_data["home_players"][i].append(1)
            stan_data["home_players_minutes"][i].append(0)
        while len(stan_data["away_players"][i]) < num_players_per_game:
            stan_data["away_players"][i].append(1)
            stan_data["away_players_minutes"][i].append(0)

    return stan_data, players_mapping, players_stats


def _load_stan_model_code(model_name: str) -> str:
    """
    Loads the Stan code for a player level model.

    Args:
        model_name: Name of the Stan model (without extension).

    Returns:
        Stan code as string.
    """
    file_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "models", "player_level", f"{model_name}.stan"
    )

    with open(file_path, encoding="utf-8") as f:
        return f.read()


def get_players_posterior_samples(
    seasons: list[int],
    model_name: str,
    chains: int = 4,
    iter_warmup: int = 2_500,
    iter_sampling: int = 2_500,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, PlayerStats]]:
    """
    Executes a player level Stan model and returns the posterior samples
    along with player metadata.

    Args:
        seasons: List of years of the seasons to be processed.
        model_name: Name of the Stan model to be used (ex: "poisson_4").
        chains: Number of MCMC chains. Default: 4.
        iter_warmup: Number of warmup iterations per chain. Default: 2500.
        iter_sampling: Number of sampling iterations per chain. Default: 2500.

    Returns:
        Tuple containing:
            - draws: DataFrame with the posterior samples.
            - players_mapping: Dictionary mapping player identifier to index.
            - players_stats: Dictionary with detailed statistics for each player,
              including: name, teams, total_minutes, total_games, total_wins,
              total_draws, and weighted_score.

    Raises:
        FileNotFoundError: If the season data or the model is not found.
        RuntimeError: If there is an error in the execution of the Stan model.
    """
    stan_data, players_mapping, players_stats = _process_seasons_data(seasons)
    stan_code = _load_stan_model_code(model_name)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".stan", delete=False
    ) as f:
        f.write(stan_code)
        stan_file_path = f.name

    try:
        model = cmdstanpy.CmdStanModel(stan_file=stan_file_path)
        fit = model.sample(
            data=stan_data,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            show_progress=True,
        )
        info = fit.summary()
        print(info[["ESS_bulk", "ESS_tail", "R_hat"]].describe())
        draws = fit.draws_pd()
    finally:
        os.unlink(stan_file_path)

    players_stats_dict = {k: v.to_dict() for k, v in players_stats.items()}
    players_stats_df = pd.DataFrame(players_stats_dict) \
        .T \
        .reset_index() \
        .rename(columns={"index": "player_id"})

    return draws, players_mapping, players_stats_df


def process_players_data(
    draws: pd.DataFrame,
    players_mapping: dict[str, int],
    players_stats: pd.DataFrame,
    both_strengths: bool = False,
) -> pd.DataFrame:
    """
    Processes the players data and returns the summary statistics.

    Args:
        draws: DataFrame with the posterior samples.
        players_mapping: Dictionary mapping player identifier to index.
        players_stats: DataFrame with the statistics for each player.
    """
    reversed_players_mapping = {str(v): k for k, v in players_mapping.items()}
    drop_cols = [col for col in draws.columns if col.startswith("raw_")]
    draws.drop(
        columns=drop_cols + [
            "chain__", "iter__", "draw__", "lp__", "accept_stat__", "stepsize__",
            "treedepth__", "n_leapfrog__", "divergent__", "energy__", "log_lik"
        ],
        inplace=True
    )

    both_strengths = draws.columns[-1].startswith("beta[")
    renames = {}
    for col in draws.columns:
        if col == "nu":
            renames[col] = "home_effect"
            continue

        if col.startswith("alpha["):
            player_id = col.replace("alpha[", "").replace("]", "")
            if both_strengths:
                renames[col] = reversed_players_mapping[player_id] + " (atk)"
            else:
                renames[col] = reversed_players_mapping[player_id]
            continue

        if col.startswith("beta["):
            player_id = col.replace("beta[", "").replace("]", "")
            renames[col] = reversed_players_mapping[player_id] + " (def)"
            continue

    draws.rename(columns=renames, inplace=True)
    if both_strengths:
        new_cols = {
            player: draws[f"{player} (atk)"] - draws[f"{player} (def)"]
            for player in players_mapping.keys()
        }

        draws = pd.concat([draws, pd.DataFrame(new_cols)], axis=1)

    summary = draws.describe(
        percentiles=[0.01, 0.025, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.975, 0.99]
    ).T
    summary.drop(columns=["count"], inplace=True)
    summary["player_id"] = summary.index.str.replace(" (atk)", "").str.replace(" (def)", "")
    summary.reset_index(inplace=True)
    summary = summary.merge(players_stats, on="player_id", how="left")

    return summary


def run_player_models(
    seasons: list[int],
    models: list[str],
    chains: int = 4,
    iter_warmup: int = 2_500,
    iter_sampling: int = 7_500,
) -> None:
    """
    Runs player level models and saves the results to CSV files.

    Args:
        seasons: List of years of the seasons to be processed.
        models: List of model names to run.
        chains: Number of MCMC chains.
        iter_warmup: Number of warmup iterations per chain.
        iter_sampling: Number of sampling iterations per chain.
    """
    season_name = str(seasons[0]) if len(seasons) == 1 else f"{seasons[0]}-{seasons[-1]}"
    print(f"Processing {season_name} seasons")

    results_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "real_data", "player_level_results"
    )
    os.makedirs(results_path, exist_ok=True)

    for model_name in models:
        draws, players_mapping, players_stats = get_players_posterior_samples(
            seasons=seasons,
            model_name=model_name,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
        )

        summary = process_players_data(
            draws,
            players_mapping,
            players_stats,
        )

        summary.to_csv(
            os.path.join(results_path, f"players_summary_{season_name}_{model_name}.csv"),
            index=False,
            encoding="utf-8"
        )


if __name__ == "__main__":
    CONFIGS = [
        {
            "seasons": [*range(2025, 2026)],
            "models": ["poisson_7"],
        },
        {
            "seasons": [*range(2020, 2026)],
            "models": ["poisson_7"],
        },
    ]

    for config in CONFIGS:
        run_player_models(**config)  # type: ignore[arg-type]
