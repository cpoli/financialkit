"""Tests for the Sharpe, Sortino and information ratios, Grinold's law and out-of-sample backtests."""

import numpy as np
import pytest

from financialkit.portfolio import (
    downside_deviation,
    equal_weight,
    fundamental_law,
    information_ratio,
    out_of_sample_returns,
    sharpe_ratio,
    sharpe_ratio_se,
    simulate_active_management,
    sortino_ratio,
)


def test_sharpe_ratio_and_lo_standard_error():
    r = np.array([0.02, -0.01, 0.03, 0.0])
    assert sharpe_ratio(r, 0.005, 12) == pytest.approx((r.mean() - 0.005) / r.std(ddof=1) * np.sqrt(12))
    np.testing.assert_allclose(sharpe_ratio(np.column_stack([r, 2 * r])), sharpe_ratio(r))
    # Lo (2002): the spread of estimated ratios matches the iid standard error.
    rng = np.random.default_rng(0)
    R = rng.normal(0.01, 0.05, size=(120, 4000))
    srs = sharpe_ratio(R)
    se = sharpe_ratio_se(R[:, 0])
    assert srs.std(ddof=1) == pytest.approx(se, rel=0.05)


def test_downside_deviation_and_sortino():
    r = np.array([0.03, -0.04, 0.01, -0.02])
    assert downside_deviation(r) == pytest.approx(np.sqrt((0.04**2 + 0.02**2) / 4))
    assert downside_deviation(r, -0.03) == pytest.approx(0.01 / 2)
    assert sortino_ratio(r, 0.0, 4) == pytest.approx(r.mean() * 4 / (downside_deviation(r) * 2))
    # For normal returns, the semideviation about the mean is sigma / sqrt(2).
    x = np.random.default_rng(1).normal(0.0, 0.1, 400_000)
    assert downside_deviation(x, 0.0) == pytest.approx(0.1 / np.sqrt(2), rel=5e-3)
    # Skew matters: a series with only small losses has a higher Sortino than Sharpe ratio.
    skewed = np.where(np.arange(100) % 10 == 0, 0.2, -0.01)
    assert sortino_ratio(skewed) > sharpe_ratio(skewed)


def test_information_ratio():
    rng = np.random.default_rng(2)
    b = rng.normal(0.01, 0.04, 240)
    a = rng.normal(0.002, 0.01, 240)
    assert information_ratio(b + a, b, 12) == pytest.approx(sharpe_ratio(a, 0.0, 12))
    assert information_ratio(np.column_stack([b + a, b + a]), b).shape == (2,)


def test_fundamental_law_by_simulation():
    """Grinold (1989): the realized IR matches IC sqrt(BR) / sqrt(1 + IC^2) within a few standard errors."""
    assert fundamental_law(0.05, 400) == pytest.approx(1.0)
    res = simulate_active_management(0.1, 100, 5000, rng=0)
    exact = 0.1 * np.sqrt(100) / np.sqrt(1 + 0.1**2)
    assert abs(res.ir - exact) < 4 * res.ir_se
    assert res.predicted_ir == pytest.approx(1.0)
    assert res.realized_ic == pytest.approx(0.1, abs=0.01)


def test_out_of_sample_and_equal_weight():
    assert equal_weight(3).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        equal_weight(0)
    R = np.random.default_rng(3).normal(0.01, 0.05, (60, 4))
    res = out_of_sample_returns(R, lambda past: equal_weight(4), 24, "1/N")
    np.testing.assert_allclose(res.returns, R[24:].mean(axis=1))
    assert res.turnover == 0.0 and res.name == "1/N"
    assert res.sharpe == pytest.approx(sharpe_ratio(res.returns))
    with pytest.raises(ValueError):
        out_of_sample_returns(R, lambda past: equal_weight(4), 60)


def test_out_of_sample_mean_is_the_mean_realized_return():
    R = np.random.default_rng(4).normal(0.01, 0.05, (60, 4))
    res = out_of_sample_returns(R, lambda past: equal_weight(4), 24)
    assert res.mean == pytest.approx(R[24:].mean())
