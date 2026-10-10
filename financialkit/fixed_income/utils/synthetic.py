"""Synthetic yield-curve data for tests and gallery examples (no market data is bundled)."""

from __future__ import annotations

import numpy as np

__all__ = ["simulate_yield_changes"]


def simulate_yield_changes(
    tenors,
    n_obs: int,
    factor_vols=(0.0060, 0.0030, 0.0015),
    noise_vol: float = 0.0004,
    tau: float = 2.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Daily zero-yield changes driven by level, slope and curvature shocks.

    Each day's change is :math:`\\sum_k \\sigma_k z_k \\ell_k(t) + \\varepsilon(t)`
    with Nelson-Siegel loadings :math:`\\ell_k` (flat, decaying, humped) and
    independent maturity-specific noise, which is the factor structure
    Litterman and Scheinkman found in Treasury returns.

    Parameters
    ----------
    tenors : array_like, shape (n,)
        Maturities (years).
    n_obs : int
        Number of days.
    factor_vols : tuple of float
        Daily standard deviations of the level, slope and curvature shocks.
    noise_vol : float
        Daily standard deviation of the idiosyncratic noise.
    tau : float
        Nelson-Siegel decay parameter of the loadings.
    rng : numpy.random.Generator, int or None

    Returns
    -------
    ndarray, shape (n_obs, n)
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(tenors, dtype=np.float64) / tau
    slope = -np.expm1(-x) / x
    loadings = np.stack([np.ones_like(x), slope, slope - np.exp(-x)])
    z = rng.standard_normal((n_obs, 3)) * np.asarray(factor_vols, dtype=np.float64)
    return z @ loadings + noise_vol * rng.standard_normal((n_obs, x.size))
