"""Tests for Lucas's tree, Breeden's CCAPM, Hansen-Singleton, Mehra-Prescott, Campbell-Cochrane and Bansal-Yaron."""

import numpy as np
import pytest

from financialkit.asset_pricing import (
    BansalYaron,
    CampbellCochrane,
    LucasTree,
    MehraPrescott,
    ccapm_expected_excess_return,
    estimate_beta,
    hansen_singleton,
    implied_risk_aversion,
)


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


def test_mehra_prescott_sharpe_ratio_obeys_hansen_jagannathan():
    """Hansen-Jagannathan (1991): |E[R - Rf]| / sigma(R) <= sigma(m) / E[m]."""
    m = MehraPrescott(beta=0.99, gamma=10.0)
    mo = m.moments()
    assert mo.sharpe_ratio == pytest.approx(mo.equity_premium / mo.equity_vol)
    assert 0.0 < mo.sharpe_ratio <= mo.sdf_vol / mo.sdf_mean
    assert repr(m) == "MehraPrescott(beta=0.99, gamma=10.0)"


def test_hansen_singleton_recovers_beta_and_gamma():
    p = LucasTree(0.97, 4.0, 0.02, 0.08).simulate(2000, rng=3)
    R = np.column_stack([p.equity_return, p.risk_free])
    Z = np.column_stack([np.ones(1999), p.growth[:-1]])
    res = hansen_singleton(p.growth[1:], R[1:], Z)
    assert np.all(np.abs(res.params - [0.97, 4.0]) < 3 * res.std_errors)
    assert res.df == 2 and res.j_pvalue > 0.01


def test_hansen_singleton_checks_row_counts():
    with pytest.raises(ValueError, match="same number of rows"):
        hansen_singleton(np.ones(5), np.ones(4))


def test_campbell_cochrane_calibration():
    """S-bar = 0.057 and S_max = 0.094 (Campbell and Cochrane 1999, Table 1)."""
    cc = CampbellCochrane()
    assert cc.S_bar == pytest.approx(0.057, abs=5e-4)
    assert np.exp(cc.s_max) == pytest.approx(0.094, abs=5e-4)
    assert cc.sensitivity(cc.s_max + 0.1) == 0.0


def test_campbell_cochrane_riskless_rate_is_constant():
    cc = CampbellCochrane()
    below = cc.grid <= cc.s_max
    np.testing.assert_allclose(np.log(cc._rf[below]) * 12, 0.0094, rtol=1e-9)


def test_campbell_cochrane_euler_equations_and_countercyclical_risk():
    cc = CampbellCochrane()
    p = cc.simulate(240_000, rng=4)
    for R in (p.equity_return, p.risk_free):
        x = p.sdf * R
        assert abs(x.mean() - 1.0) < 4 * x.std() / np.sqrt(x.size)
    s = np.linspace(cc.s_bar - 3, cc.s_max, 50)
    assert np.all(np.diff(cc.price_consumption(s)) > 0)
    assert np.all(np.diff(cc.max_sharpe_ratio(s)) < 0)
    np.testing.assert_array_equal(cc.surplus_path(100, rng=5), cc._simulate(100, 5)[0])


def test_campbell_cochrane_premium_beats_lucas():
    m = CampbellCochrane().moments(n_years=2000)
    assert m.equity_premium > 0.05
    assert m.equity_premium > 50 * LucasTree(0.99, 2.0, 0.0189, 0.015).moments().equity_premium


def test_campbell_cochrane_rejects_bad_parameters():
    with pytest.raises(ValueError, match="phi"):
        CampbellCochrane(phi=1.0)
    with pytest.raises(ValueError, match="sigma"):
        CampbellCochrane(sigma=0.0)


def test_bansal_yaron_loadings():
    """A1 = (1 - 1/psi) / (1 - kappa1 rho) and A1m = (Phi - 1/psi) / (1 - kappa1m rho) (Bansal and Yaron 2004, eqs. 8 and 11)."""
    by = BansalYaron()
    assert by.A1 == pytest.approx((1 - 1 / 1.5) / (1 - by.kappa1 * 0.979))
    assert by.A1m == pytest.approx((3.0 - 1 / 1.5) / (1 - by.kappa1_m * 0.979))
    assert by.A2 < 0 and by.lambda_e > 0 and by.beta_e > 0


@pytest.mark.parametrize("psi", [1.5, 0.5])
def test_bansal_yaron_euler_equations(psi):
    """E[m R] = 1 for the market and the bill: checks every log-linear coefficient."""
    p = BansalYaron(psi=psi).simulate(12 * 20_000, rng=6)
    for R in (p.equity_return, p.risk_free):
        x = p.sdf * R
        assert abs(x.mean() - 1.0) < 4 * x.std() / np.sqrt(x.size)


def test_bansal_yaron_premium_matches_simulation():
    by = BansalYaron()
    p = by.simulate(12 * 20_000, rng=7)
    excess = np.log(p.equity_return / p.risk_free)
    assert abs(excess.mean() - by.equity_premium()) < 4 * excess.std() / np.sqrt(excess.size)
    assert by.risk_free_rate([0.0, 0.001]).shape == (2,)
    assert by.equity_premium([by.sigma**2, 2 * by.sigma**2]).shape == (2,)
    m = by.moments(n_years=500)
    assert m.equity_premium > 0.03 and m.price_dividend > 10


def test_bansal_yaron_rejects_bad_parameters():
    with pytest.raises(ValueError, match="psi"):
        BansalYaron(psi=1.0)
    with pytest.raises(ValueError, match="rho and nu"):
        BansalYaron(rho=1.0)
    with pytest.raises(ValueError, match="no stationary"):
        BansalYaron(beta=1.0, mu=0.01)
