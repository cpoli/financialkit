"""Tests for Parkinson's range estimator, realized variance, the Mincer-Zarnowitz regression and the HAR model."""

import math

import numpy as np
import pytest

from financialkit.constants import TRADING_DAYS_PER_YEAR
from financialkit.time_series import GARCH, close_to_close_variance, fit_har, har_regressors, mincer_zarnowitz, ols, parkinson_variance, realized_variance
from financialkit.time_series.utils import simulate_intraday_returns, simulate_ohlc


def test_parkinson_is_unbiased_and_about_five_times_more_efficient():
    """E[(ln H/L)^2] = 4 ln 2 sigma^2 dt for Brownian motion (Parkinson 1980, eq. (4)); the
    estimator's variance is about 1/5.2 of the squared return's."""
    sigma = 0.3
    ohlc = simulate_ohlc(1000, sigma, steps_per_day=2000, rng=0)
    daily = sigma**2 / TRADING_DAYS_PER_YEAR
    pk = parkinson_variance(ohlc.high, ohlc.low)
    cc = close_to_close_variance(np.r_[ohlc.open[0], ohlc.close])
    # Discrete sampling of the high and low biases the range by about 1% at this grid,
    # well inside the 4 standard-error band.
    assert abs(pk.mean() - daily) < 4 * pk.std() / np.sqrt(pk.size)
    assert abs(cc.mean() - daily) < 4 * cc.std() / np.sqrt(cc.size)
    assert 3.5 < cc.var() / pk.var() < 7.5
    assert np.all(ohlc.high >= np.maximum(ohlc.open, ohlc.close)) and np.all(ohlc.low <= np.minimum(ohlc.open, ohlc.close))


def test_parkinson_closed_form_and_validation():
    assert parkinson_variance([math.e], [1.0])[0] == pytest.approx(1 / (4 * math.log(2)))
    with pytest.raises(ValueError):
        parkinson_variance([1.0], [2.0])


def test_realized_variance_is_unbiased_for_daily_variance():
    v = GARCH(1e-6, 0.08, 0.9).simulate(2000, rng=1).variance
    intraday = simulate_intraday_returns(v, 78, rng=2)
    rv = realized_variance(intraday)
    assert rv.shape == (2000,)
    diff = rv - v
    assert abs(diff.mean()) < 4 * diff.std() / np.sqrt(diff.size)
    np.testing.assert_allclose(realized_variance(intraday.T, axis=0), rv)


def test_mincer_zarnowitz_r_squared_rv_beats_squared_returns():
    """With the true variance as forecast, R^2 against r^2 is bounded near 1/3 for normal
    innovations (Andersen and Bollerslev 1998, eq. (8)) but close to 1 against realized variance."""
    v = GARCH(1e-6, 0.1, 0.88).simulate(3000, rng=3).variance
    intraday = simulate_intraday_returns(v, 288, rng=4)
    r2 = intraday.sum(axis=1) ** 2
    mz_r2 = mincer_zarnowitz(r2, v)
    mz_rv = mincer_zarnowitz(realized_variance(intraday), v)
    assert mz_r2.r_squared < 0.45 < 0.9 < mz_rv.r_squared
    assert mz_rv.coef[1] == pytest.approx(1.0, abs=4 * mz_rv.std_errors[1])


def test_har_regressors_are_trailing_means():
    x = np.random.default_rng(10).random(40)
    X = har_regressors(x)
    for i, t in enumerate(range(21, 40)):
        np.testing.assert_allclose(X[i], [x[t], x[t - 4 : t + 1].mean(), x[t - 21 : t + 1].mean()], rtol=1e-12)
    with pytest.raises(ValueError):
        har_regressors(x[:22])


def test_har_fit_predict_and_forecast_are_one_regression():
    rng = np.random.default_rng(11)
    n = 6000
    rv = np.empty(n)
    rv[:22] = 1.0
    true = np.array([0.1, 0.35, 0.3, 0.25])
    for t in range(22, n):
        rv[t] = true[0] + true[1] * rv[t - 1] + true[2] * rv[t - 5 : t].mean() + true[3] * rv[t - 22 : t].mean() + 0.1 * rng.standard_normal()
    res = fit_har(rv)
    X = har_regressors(rv)
    direct = ols(rv[22:], np.column_stack([np.ones(n - 22), X[:-1]]))
    np.testing.assert_allclose(res.coef, direct.coef, rtol=1e-12)
    assert np.all(np.abs(res.coef - true) < 4 * res.ols.std_errors)
    pred = res.predict(rv)
    assert np.all(np.isnan(pred[:22]))
    np.testing.assert_allclose(pred[22:], rv[22:] - direct.residuals, rtol=1e-10)
    assert res.forecast() == pytest.approx(res.coef[0] + X[-1] @ res.coef[1:])
    assert fit_har(rv, horizons=(1, 10)).coef.size == 3
