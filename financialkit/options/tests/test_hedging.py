"""Tests for the discrete delta-hedging simulation."""

import numpy as np
import pytest

from financialkit.options import black_scholes_price, delta_hedge


@pytest.mark.parametrize("option_type,q", [("call", 0.0), ("put", 0.03)])
def test_risk_neutral_hedge_pnl_has_zero_mean(option_type, q):
    """The discounted hedge portfolio is a Q-martingale, so E[P&L] = 0 for any n."""
    res = delta_hedge(100.0, 105.0, 1.0, 0.05, 0.25, n_rebalance=12, n_paths=20_000, q=q, option_type=option_type, rng=3)
    se = res.pnl.std() / np.sqrt(res.pnl.size)
    assert abs(res.pnl.mean()) < 4 * se
    assert res.option_price == pytest.approx(black_scholes_price(100.0, 105.0, 1.0, 0.05, 0.25, q, option_type))


def test_hedge_error_shrinks_like_one_over_sqrt_n():
    """Boyle and Emanuel (1980): std of the hedging error ~ 1/sqrt(n)."""
    std = [delta_hedge(100.0, 100.0, 0.5, 0.03, 0.2, n, 4000, mu=0.1, rng=7).pnl.std() for n in (16, 64)]
    # quadrupling n halves the std; 15% slack covers sampling noise and O(1/n) terms
    assert std[0] / std[1] == pytest.approx(2.0, rel=0.15)


def test_shapes():
    res = delta_hedge(100.0, 100.0, 1.0, 0.05, 0.2, 10, 50, rng=0)
    assert res.spot.shape == (50, 11) and res.pnl.shape == (50,) and res.times.shape == (11,)
