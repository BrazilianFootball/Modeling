"""
Real data analysis package.

This package provides tools for analyzing real football data using
statistical models and simulations.
"""

from .core import (
    process_data,
    main,
    run_model_with_real_data,
    set_team_strengths,
    run_real_data_model,
)

from .data import (
    generate_all_matches_data,
    generate_real_data_stan_input,
    load_all_matches_data,
    load_real_data,
    check_results_exist,
)

from .evaluation import (
    brier_score,
    log_score,
    ranked_probability_score,
    interval_score,
    calculate_metrics,
)

from .simulation import (
    get_real_points_evolution,
    simulate_competition,
    update_probabilities,
    calculate_final_positions_probs,
)

from .visualization import (
    generate_quantiles,
    generate_points_evolution_by_team,
    generate_boxplot,
)

__all__ = [
    # Core functions
    'process_data',
    'main',
    'run_model_with_real_data',
    'set_team_strengths',
    'run_real_data_model',

    # Data functions
    'generate_all_matches_data',
    'generate_real_data_stan_input',
    'load_all_matches_data',
    'load_real_data',
    'check_results_exist',

    # Evaluation functions
    'brier_score',
    'log_score',
    'ranked_probability_score',
    'interval_score',
    'calculate_metrics',

    # Simulation functions
    'get_real_points_evolution',
    'simulate_competition',
    'update_probabilities',
    'calculate_final_positions_probs',

    # Visualization functions
    'generate_quantiles',
    'generate_points_evolution_by_team',
    'generate_boxplot',
]
