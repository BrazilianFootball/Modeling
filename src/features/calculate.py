# Based on https://github.com/hyunjimoon/SBC/blob/master/R/calculate.R

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binom, hypergeom


def adjust_gamma(
    n: int,
    l: int,  # noqa: E741
    k: int | None = None,
    conf_level: float = 0.95,
) -> float:
    """Adjusts gamma parameter for simultaneous confidence intervals."""
    if k is None:
        k = n

    if any(x < 1 for x in [k, n, l]):
        raise ValueError("Parameters 'n', 'l' and 'k' must be positive integers.")
    if conf_level >= 1 or conf_level <= 0:
        raise ValueError("Value of 'conf_level' must be in (0,1).")

    if l == 1:
        return adjust_gamma_optimize(n, k, conf_level)

    return adjust_gamma_simulate(n, l, k, conf_level)


def adjust_gamma_optimize(n: int, k: int, conf_level: float = 0.95) -> float:
    """Optimizes gamma for a single set of samples."""

    def target(gamma: float) -> float:
        """Objective function for optimization."""
        lims = ecdf_intervals(n, 1, k, gamma)
        coverage = np.mean((lims["upper"] - lims["lower"]) / n)
        return abs(coverage - conf_level)

    result = minimize_scalar(target, bounds=(0, 1), method="bounded")
    return result.x


def adjust_gamma_simulate(
    n: int,
    l: int,  # noqa: E741
    k: int,
    conf_level: float = 0.95,
    m: int = 5000,
) -> float:
    """Adjusts gamma via simulation for multiple chains."""
    gamma = np.zeros(m)
    z = np.linspace(1 / k, 1 - 1 / k, k - 1)

    for i in range(m):
        u = np.random.uniform(0, 1, (l, n))
        scaled_ecdfs = (u[:, :, None] <= z).mean(axis=1)
        gamma[i] = 2 * min(
            np.min(binom.cdf(scaled_ecdfs, n, z)),
            np.min(binom.sf(scaled_ecdfs - 1, n, z)),
        )

    return np.quantile(gamma, 1 - conf_level)


def ecdf_intervals(n: int, l: int, k: int, gamma: float) -> dict[str, np.ndarray]:  # noqa: E741
    """Calculates confidence intervals for ECDF."""
    z = np.linspace(0, 1, k + 1)
    lims = {}

    if l == 1:
        lims["lower"] = binom.ppf(gamma / 2, n, z) / n
        lims["upper"] = binom.ppf(1 - gamma / 2, n, z) / n
    else:
        n = n * (l - 1)
        k = np.floor(z * l * n)
        lims["lower"] = hypergeom.ppf(gamma / 2, n + n, n, k) / n
        lims["upper"] = hypergeom.ppf(1 - gamma / 2, n + n, n, k) / n

    lims["lower"] = np.repeat(lims["lower"][:-1], 2)
    lims["upper"] = np.repeat(lims["upper"][:-1], 2)

    return lims
