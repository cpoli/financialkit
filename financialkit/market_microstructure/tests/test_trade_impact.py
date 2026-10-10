"""Tests for Hasbrouck's trade-impact VAR."""

import numpy as np
import pytest

from financialkit.market_microstructure import hasbrouck_var


def _structural(n, lam, rho, phi, rng):
    """Signed flow x_t = rho x_{t-1} + u_t; efficient price moves lam u_t; quotes add a transitory phi x_t."""
    u = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = u[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + u[t]
    m = np.cumsum(lam * u + 0.05 * rng.standard_normal(n))
    q = m + phi * x
    return np.diff(q), x[1:]


def test_permanent_impact_recovers_the_information_content():
    """Hasbrouck (1991), Sec. II: the cumulative response to a trade innovation converges to its permanent effect."""
    rng = np.random.default_rng(0)
    lam, reps = 0.1, 20
    est = np.array([hasbrouck_var(*_structural(5000, lam, 0.5, 0.03, rng), lags=8, horizon=30).permanent_impact for _ in range(reps)])
    assert abs(est.mean() - lam) < 4 * est.std(ddof=1) / np.sqrt(reps)


def test_trade_share_and_shapes():
    r, x = _structural(4000, 0.1, 0.0, 0.0, np.random.default_rng(1))
    res = hasbrouck_var(r, x, lags=3, horizon=10)
    assert res.impulse.shape == (11,) and res.coef_r.shape == (7,) and res.coef_x.shape == (6,)
    # Trade innovations carry variance 0.01 against 0.0025 of public news: share 0.8.
    assert res.trade_share == pytest.approx(0.8, abs=0.03)
    with pytest.raises(ValueError):
        hasbrouck_var(r[:5], x[:5], lags=3)
