"""Tests for Lucas's tree, Breeden's CCAPM and the Mehra-Prescott economy."""

import numpy as np
import pytest

from financialkit.asset_pricing import LucasTree, MehraPrescott, ccapm_expected_excess_return, estimate_beta, implied_risk_aversion


def test_lucas_log_premium_is_gamma_sigma_squared():
    for g in (1.0, 2.0, 5.0, 10.0):
        m = LucasTree(0.98, g, 0.02, 0.04).moments()
        assert np.log(m.equity_return) - np.log(m.risk_free) == pytest.approx(g * 0.04**2)


def test_lucas_log_utility_price_dividend():
    """With log utility the price-dividend ratio is beta / (1 - beta), whatever the growth process (Lucas 1978)."""
    assert LucasTree(0.96, 1.0, 0.05, 0.2).moments().price_dividend == pytest.approx(0.96 / 0.04)


@pytest.mark.parametrize("economy", [LucasTree(0.99, 4.0), MehraPrescott(0.99, 4.0)])
def test_euler_equations_hold(economy):
    """E[m R] = 1 for the equity and the bond, checked by simulation."""
    p = economy.simulate(200_000, rng=0)
    for R in (p.equity_return, p.risk_free):
        x = p.sdf * R
        assert abs(x.mean() - 1.0) < 4 * x.std() / np.sqrt(x.size)


def test_lucas_moments_match_simulation():
    tree = LucasTree(0.99, 3.0, 0.02, 0.05)
    m = tree.moments()
    p = tree.simulate(200_000, rng=1)
    se = p.equity_return.std() / np.sqrt(p.growth.size)
    assert abs(p.equity_return.mean() - m.equity_return) < 4 * se
    assert p.equity_return.std() == pytest.approx(m.equity_vol, rel=0.01)
    assert p.sdf.std() == pytest.approx(m.sdf_vol, rel=0.02)
    assert m.sdf_mean * m.risk_free == pytest.approx(1.0)


def test_mehra_prescott_calibration():
    """Mehra and Prescott (1985, Sec. 3): mean growth 1.018, std 0.036, autocorrelation -0.14."""
    mp = MehraPrescott(0.99, 2.0)
    G = mp.growth
    assert mp.stationary @ G == pytest.approx(1.018)
    assert np.sqrt(mp.stationary @ (G - 1.018) ** 2) == pytest.approx(0.036)
    assert mp.growth_autocorr == pytest.approx(-0.14)


def test_mehra_prescott_iid_case_has_constant_price_dividend():
    mp = MehraPrescott(0.98, 3.0, phi=0.5)
    k = 0.98 * np.mean(mp.growth**-2.0)
    np.testing.assert_allclose(mp.price_dividend, k / (1 - k))
    np.testing.assert_allclose(mp.risk_free, 1 / (0.98 * np.mean(mp.growth**-3.0)))


def test_equity_premium_puzzle():
    """Mehra and Prescott (1985, Fig. 4): with gamma <= 10, beta < 1 and a riskless rate in [0, 4%],
    the premium is at most about 0.35%, against 6.18% in the data."""
    best = 0.0
    for g in np.linspace(0.1, 10, 60):
        for b in np.linspace(0.8, 0.9999, 120):
            try:
                m = MehraPrescott(b, g).moments()
            except ValueError:
                continue
            if 0 <= m.risk_free - 1 <= 0.04:
                best = max(best, m.equity_premium)
    assert 0.003 < best < 0.004


def test_breeden_ccapm_exact_for_lucas_log_returns():
    gamma, sigma = 5.0, 0.04
    tree = LucasTree(0.99, gamma, 0.02, sigma)
    p = tree.simulate(100_000, rng=2)
    bc = estimate_beta(np.log(p.equity_return), np.log(p.growth))
    assert bc == pytest.approx(1.0)
    m = tree.moments()
    assert ccapm_expected_excess_return(bc, sigma, gamma) == pytest.approx(np.log(m.equity_return / m.risk_free))


def test_implied_risk_aversion_and_bad_parameters():
    assert implied_risk_aversion(0.06, 0.2, 0.03, correlation=0.5) == pytest.approx(20.0)
    with pytest.raises(ValueError):
        LucasTree(1.0, 0.5, 0.05, 0.02)
    with pytest.raises(ValueError):
        LucasTree(1.2, 2.0)
    with pytest.raises(ValueError):
        MehraPrescott(0.99, -1.0)
    with pytest.raises(ValueError):
        MehraPrescott(0.99, 2.0, phi=1.0)
