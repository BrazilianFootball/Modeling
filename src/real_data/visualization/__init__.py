"""
Visualization module for real data analysis.

This module contains functions for generating plots and visualizations
of the analysis results.
"""

from .plots import (
    generate_quantiles,
    generate_points_evolution_by_team,
    generate_boxplot,
)

__all__ = [
    'generate_quantiles',
    'generate_points_evolution_by_team',
    'generate_boxplot',
]
