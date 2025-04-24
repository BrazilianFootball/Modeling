# Based on https://github.com/hyunjimoon/SBC/blob/master/R/calculate.R

import numpy as np

from typing import Dict
from scipy.stats import binom, hypergeom
from scipy.optimize import minimize_scalar


def adjust_gamma(N: int, L: int, K: int = None, conf_level: float = 0.95) -> float:
    """Adjusts gamma parameter for simultaneous confidence intervals."""
    if K is None:
        K = N

    if any(x < 1 for x in [K, N, L]):
        raise ValueError("Parameters 'N', 'L' and 'K' must be positive integers.")
    if conf_level >= 1 or conf_level <= 0:
        raise ValueError("Value of 'conf_level' must be in (0,1).")

    if L == 1:
        return adjust_gamma_optimize(N, K, conf_level)
    else:
        return adjust_gamma_simulate(N, L, K, conf_level)


def adjust_gamma_optimize(N: int, K: int, conf_level: float = 0.95) -> float:
    """Optimizes gamma for a single set of samples."""

    def target(gamma: float) -> float:
        """Objective function for optimization."""
        lims = ecdf_intervals(N, 1, K, gamma)
        coverage = np.mean((lims["upper"] - lims["lower"]) / N)
        return abs(coverage - conf_level)

    result = minimize_scalar(target, bounds=(0, 1), method="bounded")
    return result.x


def adjust_gamma_simulate(
    N: int, L: int, K: int, conf_level: float = 0.95, M: int = 5000
) -> float:
    """Adjusts gamma via simulation for multiple chains."""
    gamma = np.zeros(M)
    z = np.linspace(1 / K, 1 - 1 / K, K - 1)

    for m in range(M):
        u = np.random.uniform(0, 1, (L, N))
        scaled_ecdfs = (u[:, :, None] <= z).mean(axis=1)
        gamma[m] = 2 * min(
            np.min(binom.cdf(scaled_ecdfs, N, z)),
            np.min(binom.sf(scaled_ecdfs - 1, N, z)),
        )

    return np.quantile(gamma, 1 - conf_level)


def ecdf_intervals(N: int, L: int, K: int, gamma: float) -> Dict[str, np.ndarray]:
    """Calculates confidence intervals for ECDF."""
    z = np.linspace(0, 1, K + 1)
    lims = {}

    if L == 1:
        lims["lower"] = binom.ppf(gamma / 2, N, z) / N
        lims["upper"] = binom.ppf(1 - gamma / 2, N, z) / N
    else:
        n = N * (L - 1)
        k = np.floor(z * L * N)
        lims["lower"] = hypergeom.ppf(gamma / 2, N + n, N, k) / N
        lims["upper"] = hypergeom.ppf(1 - gamma / 2, N + n, N, k) / N

    lims["lower"] = np.repeat(lims["lower"][:-1], 2)
    lims["upper"] = np.repeat(lims["upper"][:-1], 2)

    return lims
