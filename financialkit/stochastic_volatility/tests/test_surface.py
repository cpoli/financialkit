"""Tests for the SVI implied-volatility surface and local volatility built from it."""

import numpy as np
import pytest
from scipy.linalg import solve_banded

from financialkit.options import black_scholes_price, implied_volatility
from financialkit.pde_methods import black_scholes_operator
from financialkit.stochastic_volatility import HestonModel, LocalVolSurface, SSVISurface, SVIFit, SVISurface, heston_price, ssvi_surface, svi_surface

S, R, Q = 100.0, 0.03, 0.01
EXPIRIES = np.array([0.25, 0.5, 1.0])
STRIKES = np.linspace(70.0, 140.0, 15)


def _flat(w: float, T: float) -> SVIFit:
    """A slice with constant total variance ``w`` (b = 0)."""
    return SVIFit(w, 0.0, 0.0, 0.0, 0.1, T, 0.0)


@pytest.fixture(scope="module")
def heston_quotes():
    """Implied vols of a Heston model: a smile free of static arbitrage by construction."""
    model = HestonModel(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
    C = np.array([heston_price(model, S, STRIKES, T, R, Q) for T in EXPIRIES])
    K, T = np.meshgrid(STRIKES, EXPIRIES)
    return K, T, np.asarray(implied_volatility(C, S, K, T, R, Q))


@pytest.fixture(scope="module")
def heston_surface(heston_quotes):
    K, T, vol = heston_quotes
    return svi_surface(K.ravel(), T.ravel(), vol.ravel(), S, R, Q)


def test_surface_fits_heston_quotes_without_arbitrage(heston_quotes, heston_surface):
    K, T, vol = heston_quotes
    assert heston_surface.maturities.tolist() == EXPIRIES.tolist()
    # Raw SVI is not Heston's exact smile; 0.2 vol points is the fit error here.
    assert np.max(np.abs(heston_surface.implied_vol(K, T) - vol)) < 2e-3
    assert heston_surface.is_arbitrage_free()
    assert heston_surface.butterfly_arbitrage().shape == (3, 301)
    assert heston_surface.calendar_arbitrage().shape == (2, 301)


def test_total_variance_is_linear_in_T_between_slices_and_flat_vol_outside():
    s = SVISurface([_flat(0.09, 1.0), _flat(0.01, 0.25)], S=100.0, r=0.0)
    assert s.maturities.tolist() == [0.25, 1.0]
    w = s.total_variance(0.3, [0.1, 0.25, 0.625, 1.0, 2.0])
    np.testing.assert_allclose(w, [0.004, 0.01, 0.05, 0.09, 0.18], rtol=1e-14)
    assert s.implied_vol(150.0, 3.0) == pytest.approx(0.3, rel=1e-14)
    one = SVISurface([_flat(0.04, 1.0)], S=100.0, r=0.0)
    np.testing.assert_allclose(one.total_variance(0.0, [0.5, 2.0]), [0.02, 0.08], rtol=1e-14)


def test_implied_vol_uses_log_forward_moneyness(heston_surface):
    """At the forward, k = 0 on every slice."""
    for sl in heston_surface.slices:
        F = heston_surface.forward(sl.T)
        assert F == pytest.approx(S * np.exp((R - Q) * sl.T), rel=1e-15)
        assert heston_surface.implied_vol(F, sl.T) == pytest.approx(sl.implied_vol(0.0), rel=1e-12)


def test_price_is_black_scholes_at_the_surface_vol(heston_surface):
    K, T = 90.0, 0.5
    vol = heston_surface.implied_vol(K, T)
    assert heston_surface.price(K, T, "put") == pytest.approx(black_scholes_price(S, K, T, R, vol, Q, "put"), rel=1e-14)


def test_calendar_arbitrage_detected():
    s = SVISurface([_flat(0.04, 0.5), _flat(0.03, 1.0)], S=100.0, r=0.0)
    np.testing.assert_allclose(s.calendar_arbitrage([0.0, 1.0]), [[-0.01, -0.01]])
    assert not s.is_arbitrage_free()
    assert s.is_arbitrage_free(tol=0.02)


def test_butterfly_arbitrage_detected_on_vogts_slice():
    """Gatheral and Jacquier (2014), Example 3.1: Axel Vogt's raw SVI slice
    (a, b, m, rho, sigma) = (-0.0410, 0.1331, 0.3586, 0.3060, 0.4153) at T = 1
    has no calendar issue but a negative density, g < 0, near k = 0.9."""
    s = SVISurface([SVIFit(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153, 1.0, 0.0)], S=100.0, r=0.0)
    g = s.butterfly_arbitrage()[0]
    k = np.linspace(-1.5, 1.5, 301)
    assert g.min() < 0 and 0.6 < k[np.argmin(g)] < 1.2
    assert not s.is_arbitrage_free()


def test_butterfly_flags_negative_variance_and_steep_wings():
    negative = SVISurface([SVIFit(-0.01, 0.01, 0.0, 0.0, 0.1, 1.0, 0.0)], S=100.0, r=0.0)
    assert np.isneginf(negative.butterfly_arbitrage([0.0])).all()
    # b (1 + |rho|) = 2.25 breaks Lee's bound on the right wing, though g(0) > 0.
    steep = SVISurface([SVIFit(0.04, 1.5, 0.5, 0.0, 0.1, 1.0, 0.0)], S=100.0, r=0.0)
    assert steep.butterfly_arbitrage([0.0])[0, 0] > 0
    assert not steep.is_arbitrage_free(k=[0.0])


def _fill_nan_by_strike(lv: LocalVolSurface) -> LocalVolSurface:
    vol = np.array([np.interp(lv.strikes, lv.strikes[np.isfinite(row)], row[np.isfinite(row)]) for row in lv.vol])
    return LocalVolSurface(lv.strikes, lv.maturities, vol)


def _dupire_forward_pde(lv: LocalVolSurface, expiries: np.ndarray, n_K: int = 601, n_t: int = 400) -> np.ndarray:
    """Calls at ``STRIKES`` from Dupire's forward equation
    ``C_T = sigma^2 K^2 C_KK / 2 - (r - q) K C_K - q C``, ``C(K, 0) = (S - K)^+``,
    by Crank-Nicolson after four implicit steps (Rannacher)."""
    K = np.linspace(0.0, 300.0, n_K)
    C = np.maximum(S - K, 0.0)
    dt = expiries[-1] / n_t
    out = []
    for n in range(n_t):
        # The forward operator is the backward one with r and q swapped.
        lo, di, up = black_scholes_operator(K, lv(K[1:-1], (n + 0.5) * dt), Q, R)
        th = 1.0 if n < 4 else 0.5
        rhs = C[1:-1] + (1 - th) * dt * (lo * C[:-2] + di * C[1:-1] + up * C[2:])
        C[0] = S * np.exp(-Q * (n + 1) * dt)
        rhs[0] += th * dt * lo[0] * C[0]
        ab = np.zeros((3, n_K - 2))
        ab[0, 1:], ab[1], ab[2, :-1] = -th * dt * up[:-1], 1 - th * dt * di, -th * dt * lo[1:]
        C[1:-1] = solve_banded((1, 1), ab, rhs)
        if np.isclose((n + 1) * dt, expiries).any():
            out.append(np.interp(STRIKES, K, C))
    return np.array(out)


def test_local_vol_from_the_surface_reprices_the_input_options(heston_surface):
    """Local volatility from the SVI surface, run through Dupire's forward
    equation, gives back the surface's implied vols to within 0.1 vol
    points at the quoted expiries for strikes 70-130 (the 140 strike at
    3 months is worth under 0.02 and is left out)."""
    lv = _fill_nan_by_strike(heston_surface.local_vol(np.linspace(20.0, 300.0, 561), np.linspace(0.01, 1.0, 100)))
    C = _dupire_forward_pde(lv, EXPIRIES)
    T = np.broadcast_to(EXPIRIES[:, None], C.shape)
    vol = np.asarray(implied_volatility(C, S, STRIKES, T, R, Q))
    inner = STRIKES <= 130.0
    err = np.abs(vol - heston_surface.implied_vol(STRIKES, T))[:, inner]
    assert err.max() < 1e-3


def test_local_vol_of_a_flat_surface_is_flat():
    s = SVISurface([_flat(0.0625 * 0.5, 0.5), _flat(0.0625, 1.0)], S=100.0, r=0.03)
    lv = s.local_vol(np.linspace(60.0, 160.0, 101), np.linspace(0.25, 1.0, 31))
    assert np.nanmax(np.abs(lv.vol[2:-2, 20:-20] - 0.25)) < 2e-3


def test_validation():
    with pytest.raises(ValueError, match="at least one slice"):
        SVISurface([], S=100.0, r=0.0)
    with pytest.raises(ValueError, match="distinct"):
        SVISurface([_flat(0.04, 1.0), _flat(0.05, 1.0)], S=100.0, r=0.0)
    with pytest.raises(ValueError, match="same length"):
        svi_surface([90.0, 100.0], [1.0], [0.2, 0.2], 100.0, 0.0)
    with pytest.raises(ValueError, match="at least five"):
        svi_surface([90.0, 100.0, 110.0], [1.0, 1.0, 1.0], [0.2, 0.2, 0.2], 100.0, 0.0)


def test_ssvi_slice_is_raw_svi():
    """Gatheral and Jacquier (2014), Section 3: an SSVI slice is the raw SVI slice with a = theta(1 - rho^2)/2, b = theta phi/2, ..."""
    s = SSVISurface(-0.5, [0.5, 2.0], [0.02, 0.09], S=100.0, r=0.03, q=0.01, eta=1.1, gamma=0.3)
    k = np.linspace(-1.5, 1.5, 31)
    for T in (0.3, 0.5, 1.2, 2.0, 3.0):
        np.testing.assert_allclose(s.to_svi(T).total_variance(k), s.total_variance(k, T), atol=1e-15)
    # theta is the at-the-money total variance.
    np.testing.assert_allclose(s.total_variance(0.0, [0.25, 0.5, 2.0, 4.0]), [0.01, 0.02, 0.09, 0.18], rtol=1e-14)
    assert s.implied_vol(s.forward(2.0), 2.0) == pytest.approx(np.sqrt(0.09 / 2.0), rel=1e-14)


@pytest.mark.parametrize("kw", [dict(eta=1.2, gamma=0.4), dict(eta=1.25, gamma=0.5), dict(phi="heston", lam=0.4)])
def test_ssvi_conditions_agree_with_the_slice_checks(kw):
    """Parameters inside Gatheral and Jacquier's Section 4 bounds pass Durrleman's and the calendar test too."""
    s = SSVISurface(-0.6, [0.25, 0.5, 1.0, 3.0], [0.01, 0.02, 0.045, 0.15], S=100.0, r=0.0, **kw)
    assert s.is_arbitrage_free()
    assert s.to_svi_surface().is_arbitrage_free(tol=1e-12)


def test_ssvi_flags_butterfly_and_calendar_arbitrage():
    steep = SSVISurface(-0.6, [1.0], [0.04], S=100.0, r=0.0, eta=4.0, gamma=0.5)
    assert steep.arbitrage_conditions()["butterfly"].min() < 0
    assert not steep.is_arbitrage_free()
    assert not steep.to_svi_surface().is_arbitrage_free()
    falling = SSVISurface(-0.3, [0.5, 1.0], [0.04, 0.03], S=100.0, r=0.0)
    assert not falling.is_arbitrage_free()
    assert np.any(falling.to_svi_surface().calendar_arbitrage() < 0)


def test_ssvi_fit_recovers_parameters_and_stays_arbitrage_free(heston_quotes):
    true = SSVISurface(-0.6, [0.5, 1.0], [0.02, 0.045], S, R, Q, eta=1.2, gamma=0.4)
    K, T = np.meshgrid(STRIKES, [0.5, 1.0])
    fit = ssvi_surface(K.ravel(), T.ravel(), np.asarray(true.implied_vol(K, T)).ravel(), S, R, Q)
    assert fit.rmse < 1e-10
    assert (fit.rho, fit.eta, fit.gamma) == pytest.approx((-0.6, 1.2, 0.4), abs=1e-7)
    np.testing.assert_allclose(fit.theta, [0.02, 0.045], rtol=1e-8)
    K, T, vol = heston_quotes
    for phi in ("power_law", "heston"):
        hf = ssvi_surface(K.ravel(), T.ravel(), vol.ravel(), S, R, Q, phi=phi)
        assert hf.is_arbitrage_free() and hf.to_svi_surface().is_arbitrage_free(tol=1e-12)
    # SSVI has three shape parameters for the whole surface; a vol point of error is the price.
    assert ssvi_surface(K.ravel(), T.ravel(), vol.ravel(), S, R, Q).rmse < 0.01


def test_ssvi_local_vol_of_a_flat_surface():
    s = SSVISurface(0.0, [1.0], [0.0625], S=100.0, r=0.03, eta=1e-6, gamma=0.5)
    lv = s.local_vol(np.linspace(60.0, 160.0, 101), np.linspace(0.25, 1.0, 31))
    assert np.nanmax(np.abs(lv.vol[2:-2, 20:-20] - 0.25)) < 2e-3


def test_ssvi_validation():
    with pytest.raises(ValueError, match="same length"):
        SSVISurface(0.0, [1.0, 2.0], [0.04], S=100.0, r=0.0)
    with pytest.raises(ValueError, match="increasing"):
        SSVISurface(0.0, [1.0, 1.0], [0.04, 0.05], S=100.0, r=0.0)
    with pytest.raises(ValueError, match="theta must be positive"):
        SSVISurface(0.0, [1.0], [0.0], S=100.0, r=0.0)
    with pytest.raises(ValueError, match="rho must lie"):
        SSVISurface(1.0, [1.0], [0.04], S=100.0, r=0.0)
    with pytest.raises(ValueError, match="phi must be one of"):
        SSVISurface(0.0, [1.0], [0.04], S=100.0, r=0.0, phi="sabr")
    with pytest.raises(ValueError, match="0 < gamma < 1"):
        SSVISurface(0.0, [1.0], [0.04], S=100.0, r=0.0, gamma=1.0)
    with pytest.raises(ValueError, match="same length"):
        ssvi_surface([90.0, 100.0], [1.0], [0.2, 0.2], 100.0, 0.0)
    with pytest.raises(ValueError, match="phi must be one of"):
        ssvi_surface([90.0, 100.0], [1.0, 1.0], [0.2, 0.2], 100.0, 0.0, phi="x")
    with pytest.raises(ValueError, match="at least two per expiry"):
        ssvi_surface([90.0, 100.0], [1.0, 2.0], [0.2, 0.2], 100.0, 0.0)


def test_surface_base_needs_total_variance():
    from financialkit.stochastic_volatility.systems.surface import _ImpliedVarianceSurface

    with pytest.raises(NotImplementedError):
        _ImpliedVarianceSurface().total_variance(0.0, 1.0)
