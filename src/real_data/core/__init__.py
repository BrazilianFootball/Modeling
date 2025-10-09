"""
Core module for real data analysis.

This module contains the main orchestration and model execution functions.
"""

from .orchestrator import (
    process_data,
    main,
)

from .model_runner import (
    run_model_with_real_data,
    set_team_strengths,
    run_real_data_model,
)

__all__ = [
    'process_data',
    'main',
    'run_model_with_real_data',
    'set_team_strengths',
    'run_real_data_model',
]
