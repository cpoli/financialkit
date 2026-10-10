"""Tests for the Fourier pricers: COS, Lewis, Carr-Madan FFT and Heston's two probabilities."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from financialkit.options import black_scholes_price
from financialkit.stochastic_volatility import (
    BatesModel,
    BlackScholesModel,
    HestonModel,
    KouModel,
    MertonJumpModel,
    VarianceGammaModel,
    carr_madan_fft,
    carr_madan_price,
    cos_density,
    cos_price,
    cos_terms,
    heston_price,
    lewis_price,
    truncation_range,
)

MODELS = [
    BlackScholesModel(0.25),
    HestonModel(0.04, 1.5, 0.05, 0.5, -0.7),
    MertonJumpModel(0.2, 0.8, -0.1, 0.15),
    VarianceGammaModel(0.2, 0.3, -0.15),
    KouModel(0.15, 1.0, 0.4, 10.0, 5.0),
    BatesModel(0.04, 1.5, 0.05, 0.5, -0.7, 0.5, -0.1, 0.1),
]
PRICERS = [cos_price, lewis_price, heston_price, carr_madan_price]
K = np.array([70.0, 95.0, 100.0, 110.0, 140.0])


@pytest.mark.parametrize("model", MODELS, ids=lambda m: type(m).__name__)
def test_characteristic_function_is_normalised(model):
    """phi(0) = 1 and phi(-i) = E[S_T / F_T] = 1: the forward is a martingale."""
    assert model.char_func(0.0, 0.7) == pytest.approx(1.0, abs=1e-14)
    assert model.char_func(-1j, 0.7) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("pricer", PRICERS, ids=lambda f: f.__name__)
def test_black_scholes_model_reproduces_closed_form(pricer):
    m = BlackScholesModel(0.25)
    for option_type in ("call", "put"):
        exact = black_scholes_price(100.0, K, 0.75, 0.04, 0.25, 0.01, option_type)
        # The FFT is interpolated between grid strikes; the integrals are accurate to quad's tolerance.
        tol = 1e-5 if pricer is carr_madan_price else 1e-8
        assert np.asarray(pricer(m, 100.0, K, 0.75, 0.04, 0.01, option_type)) == pytest.approx(exact, abs=tol)


@pytest.mark.parametrize("model", MODELS[1:], ids=lambda m: type(m).__name__)
def test_fourier_methods_agree(model):
    ref = np.asarray(lewis_price(model, 100.0, K, 1.0, 0.03))
    # Heston's P_j integrands decay like |phi(u)|/u, slower than Lewis's |phi|/u^2, so quad stops near 1e-7.
    assert np.asarray(heston_price(model, 100.0, K, 1.0, 0.03)) == pytest.approx(ref, abs=1e-6)
    assert np.asarray(cos_price(model, 100.0, K, 1.0, 0.03)) == pytest.approx(ref, abs=1e-6)
    assert np.asarray(carr_madan_price(model, 100.0, K, 1.0, 0.03)) == pytest.approx(ref, abs=1e-4)


@pytest.mark.parametrize("model", MODELS[1:], ids=lambda m: type(m).__name__)
def test_put_call_parity(model):
    c = np.asarray(model.price(100.0, K, 0.5, 0.05, 0.02, "call"))
    p = np.asarray(model.price(100.0, K, 0.5, 0.05, 0.02, "put"))
    assert c - p == pytest.approx(100.0 * np.exp(-0.01) - K * np.exp(-0.025), abs=1e-10)


def test_heston_fang_oosterlee_reference():
    """Fang and Oosterlee (2008), Section 5.2 (Heston test): reference price 5.785155450."""
    m = HestonModel(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
    assert lewis_price(m, 100.0, 100.0, 1.0, 0.0) == pytest.approx(5.785155450, abs=1e-7)
    assert cos_price(m, 100.0, 100.0, 1.0, 0.0) == pytest.approx(5.785155450, abs=1e-6)


@pytest.mark.parametrize(("T", "ref"), [(0.1, 10.993703187), (1.0, 19.099354724)])
def test_variance_gamma_fang_oosterlee_reference(T, ref):
    """Fang and Oosterlee (2008), Table 7: VG call, S = 100, K = 90, r = 0.1, sigma = 0.12, theta = -0.14, nu = 0.2."""
    m = VarianceGammaModel(sigma=0.12, nu=0.2, theta=-0.14)
    assert lewis_price(m, 100.0, 90.0, T, 0.1) == pytest.approx(ref, abs=1e-8)
    # The VG density has a kink at short maturity, so COS converges algebraically there.
    assert cos_price(m, 100.0, 90.0, T, 0.1, N=8192) == pytest.approx(ref, abs=1e-6)


def test_cos_converges_with_more_terms():
    m = HestonModel(0.0175, 1.5768, 0.0398, 0.5751, -0.5711)
    errs = [abs(cos_price(m, 100.0, 100.0, 1.0, 0.0, N=n) - 5.785155450) for n in (32, 64, 128)]
    assert errs[0] > errs[1] > errs[2]


def test_cos_terms_grow_with_interval():
    assert cos_terms(-1.0, 1.0) == 256
    assert cos_terms(-40.0, 40.0) == 4096


def test_truncation_range_black_scholes_cumulants():
    m = BlackScholesModel(0.3)
    c1, c2, c4 = m.cumulants(2.0)
    # Finite differences with h = 0.02 in the cumulant generating function: errors O(h^2) or rounding.
    assert (c1, c2) == pytest.approx((-0.09, 0.18), abs=1e-9)
    assert abs(c4) < 1e-5
    a, b = truncation_range(m, 2.0, L=10.0)
    assert b - a == pytest.approx(20 * np.sqrt(0.18 + np.sqrt(abs(c4))))


def test_cos_density_matches_normal_and_variance_gamma():
    x = np.linspace(-0.6, 0.6, 13)
    assert cos_density(BlackScholesModel(0.2), x, 1.0) == pytest.approx(norm.pdf(x, -0.02, 0.2), abs=1e-10)
    vg = VarianceGammaModel(0.2, 0.25, -0.1)
    assert cos_density(vg, x, 1.0, N=1024) == pytest.approx(vg.pdf(x, 1.0), abs=1e-6)


def test_carr_madan_grid_and_damping():
    res = carr_madan_fft(BlackScholesModel(0.2), 100.0, 1.0, 0.05, N=4096, eta=0.25)
    assert res.strikes.size == 4096
    assert np.log(res.strikes[1] / res.strikes[0]) == pytest.approx(2 * np.pi / (4096 * 0.25))
    mid = (res.strikes > 50) & (res.strikes < 200)
    assert res.calls[mid] == pytest.approx(black_scholes_price(100.0, res.strikes[mid], 1.0, 0.05, 0.2), abs=1e-6)


def test_lewis_integrand_is_the_expected_payoff():
    """Lewis's price equals the discounted payoff integrated against the COS density."""
    m = MertonJumpModel(0.2, 1.0, -0.05, 0.1)
    F, Kx = 100.0 * np.exp(0.02), 105.0
    integral = quad(lambda x: max(F * np.exp(x) - Kx, 0.0) * float(cos_density(m, [x], 1.0)[0]), -2.0, 2.0, limit=200)[0]
    assert lewis_price(m, 100.0, Kx, 1.0, 0.02) == pytest.approx(np.exp(-0.02) * integral, abs=1e-6)


def test_invalid_method_and_option_type():
    m = BlackScholesModel(0.2)
    with pytest.raises(ValueError):
        m.price(100.0, 100.0, 1.0, 0.0, method="pde")
    with pytest.raises(ValueError):
        cos_price(m, 100.0, 100.0, 1.0, 0.0, option_type="straddle")
    with pytest.raises(ValueError):
        BlackScholesModel(0.0)


def test_implied_vol_round_trip():
    m = BlackScholesModel(0.27)
    assert np.asarray(m.implied_vol(100.0, K, 1.0, 0.03, 0.01)) == pytest.approx(0.27, abs=1e-7)
    assert m.implied_vol(100.0, 100.0, 1.0, 0.03) == pytest.approx(0.27, abs=1e-7)
