"""
Simulation module for real data analysis.

This module contains functions for simulating competitions and
calculating probabilities.
"""

from .competition import (
    get_real_points_evolution,
    simulate_competition,
    update_probabilities,
    calculate_final_positions_probs,
)

__all__ = [
    'get_real_points_evolution',
    'simulate_competition',
    'update_probabilities',
    'calculate_final_positions_probs',
]
