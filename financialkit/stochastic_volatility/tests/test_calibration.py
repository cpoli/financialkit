"""Tests for calibrating characteristic-function models to a smile."""

import numpy as np
import pytest

from financialkit.options import black_scholes_price
from financialkit.stochastic_volatility import (
    BatesModel,
    BlackScholesModel,
    FourierCalibration,
    HestonModel,
    KouModel,
    VarianceGammaModel,
    fourier_calibrate,
)

S, r, q = 100.0, 0.03, 0.01
MATURITIES = np.array([0.25, 0.5, 1.0, 2.0])
K, T = np.meshgrid(np.linspace(75, 130, 12), MATURITIES)

CASES = [
    (HestonModel, dict(v0=0.04, kappa=1.5, theta=0.06, xi=0.6, rho=-0.7), dict(v0=0.02, kappa=3.0, theta=0.03, xi=0.3, rho=-0.2)),
    (
        BatesModel,
        dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6, lam=0.5, mu_j=-0.1, delta=0.15),
        dict(v0=0.03, kappa=1.0, theta=0.05, xi=0.3, rho=-0.3, lam=0.3, mu_j=-0.05, delta=0.1),
    ),
    (VarianceGammaModel, dict(sigma=0.2, nu=0.3, theta=-0.15), dict(sigma=0.3, nu=0.1, theta=0.0)),
    (KouModel, dict(sigma=0.15, lam=1.0, p=0.3, eta1=10.0, eta2=5.0), dict(sigma=0.2, lam=0.5, p=0.5, eta1=6.0, eta2=8.0)),
]


def surface_prices(model, option_type="call"):
    return np.array([model.price(S, k, t[0], r, q, option_type) for k, t in zip(K, T, strict=True)])


@pytest.mark.parametrize(("cls", "true", "x0"), CASES, ids=[c[0].__name__ for c in CASES])
def test_round_trip_recovers_parameters(cls, true, x0):
    """Prices the model generated are refitted exactly from a distant start.

    The 1e-6 tolerance is far above the ~1e-9 the fit reaches and far below
    any parameter difference visible in the smile.
    """
    fit = fourier_calibrate(cls, x0, S, K, T, r, q, prices=surface_prices(cls(**true)))
    assert isinstance(fit, FourierCalibration) and fit.converged
    assert fit.params == pytest.approx(true, abs=1e-6)
    assert fit.rmse < 1e-6
    assert np.abs(fit.residuals).max() < 1e-8
    assert fit.strikes.shape == fit.maturities.shape == fit.model_vols.shape == (K.size,)


def test_fit_to_put_vols_with_fixed_parameter():
    true = VarianceGammaModel(sigma=0.2, nu=0.3, theta=-0.15)
    vols = np.array([true.implied_vol(S, k, t[0], r, q) for k, t in zip(K, T, strict=True)])
    fit = fourier_calibrate(VarianceGammaModel, dict(sigma=0.3, nu=0.1), S, K, T, r, q, vols=vols, option_type="put", fixed=dict(theta=-0.15))
    assert fit.params == pytest.approx(dict(sigma=0.2, nu=0.3, theta=-0.15), abs=1e-6)
    assert fit.market_vols == pytest.approx(vols.ravel())


def test_rmse_is_in_vol_points():
    """Two quotes of one option at 19% and 21%: the flat fit is 20%, one vol point off each."""
    fit = fourier_calibrate(BlackScholesModel, dict(sigma=0.3), S, [100.0, 100.0], 1.0, r, vols=[0.19, 0.21], method="lewis")
    assert fit.params["sigma"] == pytest.approx(0.2, abs=1e-4)
    assert fit.rmse == pytest.approx(1.0, rel=1e-3)
    assert fit.residuals == pytest.approx([0.01, -0.01], rel=1e-2)


class CappedBlackScholes(BlackScholesModel):
    """Rejects sigma above 0.3, so a fit to 50% vols must step into rejected parameters."""

    def __init__(self, sigma):
        if sigma > 0.3:
            raise ValueError("sigma too large")
        super().__init__(sigma)


def test_rejected_parameters_are_penalised():
    vols = np.full(5, 0.5)
    prices = black_scholes_price(S, np.linspace(80, 120, 5), 1.0, r, vols)
    fit = fourier_calibrate(CappedBlackScholes, dict(sigma=0.2), S, np.linspace(80, 120, 5), 1.0, r, prices=prices, bounds=dict(sigma=(0.01, 1.0)))
    assert 0.2 < fit.params["sigma"] <= 0.3
    assert fit.rmse > 10.0


def test_validation():
    with pytest.raises(ValueError, match="exactly one"):
        fourier_calibrate(HestonModel, CASES[0][2], S, K, T, r)
    with pytest.raises(ValueError, match="exactly one"):
        fourier_calibrate(HestonModel, CASES[0][2], S, K, T, r, prices=1.0, vols=0.2)
    with pytest.raises(ValueError, match="no parameters"):
        fourier_calibrate(HestonModel, dict(sigma=0.2), S, K, T, r, vols=0.2)
    with pytest.raises(ValueError, match="both fitted and fixed"):
        fourier_calibrate(VarianceGammaModel, dict(sigma=0.2, nu=0.3), S, K, T, r, vols=0.2, fixed=dict(nu=0.3))
    with pytest.raises(ValueError, match="infinite"):
        fourier_calibrate(VarianceGammaModel, dict(sigma=2.0, nu=4.0), S, K, T, r, vols=0.2)
