"""Tests for spread measures, the Roll and Corwin-Schultz estimators and Lee-Ready classification."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_ATOL
from financialkit.market_microstructure import (
    align_quotes,
    corwin_schultz_spread,
    effective_spread,
    lee_ready,
    midquote,
    price_impact,
    quote_rule,
    quoted_spread,
    realized_spread,
    roll_spread,
    tick_test,
)


def test_spread_decomposition():
    """Huang-Stoll (1996): effective spread = realized spread + price impact, trade by trade."""
    rng = np.random.default_rng(0)
    n = 50
    m = 100 + rng.standard_normal(n).cumsum() * 0.01
    bid, ask = m - 0.05, m + 0.05
    side = rng.choice([-1.0, 1.0], n)
    price = m + side * rng.uniform(0, 0.05, n)
    later = m + rng.standard_normal(n) * 0.02
    np.testing.assert_allclose(
        effective_spread(price, bid, ask, side), realized_spread(price, later, side) + price_impact(bid, ask, later, side), atol=DEFAULT_ATOL
    )
    assert np.all(effective_spread(price, bid, ask, side) <= quoted_spread(bid, ask) + DEFAULT_ATOL)
    np.testing.assert_allclose(effective_spread(ask, bid, ask, np.ones(n), relative=True), quoted_spread(bid, ask, relative=True))


def test_input_validation():
    with pytest.raises(ValueError):
        midquote([100.1], [100.0])
    with pytest.raises(ValueError):
        effective_spread([100.0], [99.0], [101.0], [0])
    with pytest.raises(ValueError):
        roll_spread([1.0, 2.0])
    with pytest.raises(ValueError):
        corwin_schultz_spread([1.0, 2.0], [1.5, 1.0])


def test_roll_estimator_recovers_spread():
    """Roll (1984): with random-walk value and independent bounce, cov(dp_t, dp_t-1) = -s^2/4."""
    rng = np.random.default_rng(1)
    s, n, reps = 0.10, 5000, 40
    est = np.array([roll_spread(100 + 0.02 * rng.standard_normal(n).cumsum() + 0.5 * s * rng.choice([-1.0, 1.0], n)) for _ in range(reps)])
    assert abs(est.mean() - s) < 4 * est.std(ddof=1) / np.sqrt(reps)


def test_roll_is_nan_for_positive_autocovariance():
    trend = np.cumsum(np.sin(np.arange(200) / 10.0))
    assert np.isnan(roll_spread(trend))


def test_corwin_schultz_unbiased_under_continuous_trading():
    """Corwin-Schultz (2012), Sec. II: highs at the ask and lows at the bid of a Brownian log price."""
    rng = np.random.default_rng(2)
    days, m, sigma, s = 2000, 390, 0.02, 0.01
    lp = np.cumsum(sigma / np.sqrt(m) * rng.standard_normal(days * m)).reshape(days, m)
    est = corwin_schultz_spread(np.exp(lp.max(1)) * (1 + s / 2), np.exp(lp.min(1)) * (1 - s / 2), clip=False)
    # Pairs overlap, so allow for the lag-one correlation by doubling the naive standard error.
    assert abs(est.mean() - s) < 4 * 2 * est.std(ddof=1) / np.sqrt(est.size)
    assert np.all(corwin_schultz_spread(np.exp(lp.max(1)), np.exp(lp.min(1))) >= 0)


def test_tick_quote_and_lee_ready_rules():
    p = [10.0, 10.0, 10.1, 10.1, 10.0, 10.0]
    assert tick_test(p).tolist() == [0, 0, 1, 1, -1, -1]
    b, a = [9.95] * 6, [10.05] * 6
    assert quote_rule([10.05, 10.0, 9.95, 10.0, 10.0, 10.0], b, a).tolist() == [1, 0, -1, 0, 0, 0]
    # At the midquote Lee-Ready falls back on the tick test.
    assert lee_ready([10.05, 10.0, 10.0, 10.02, 10.0, 9.99], b, a).tolist() == [1, -1, -1, 1, -1, -1]


def test_align_quotes_five_second_rule():
    b, a = align_quotes([1.0, 6.0, 12.0], [0.0, 5.0, 10.0], [1.0, 2.0, 3.0], [1.1, 2.1, 3.1], delay=5.0)
    assert np.isnan(b[0]) and b[1:].tolist() == [1.0, 2.0] and a[2] == 2.1
    with pytest.raises(ValueError):
        align_quotes([1.0], [1.0, 0.0], [1.0, 1.0], [1.1, 1.1])


def test_lee_ready_beats_tick_test_inside_the_quotes():
    """Lee-Ready (1991), Table 2: the quote rule is more accurate than the tick test when quotes are known."""
    rng = np.random.default_rng(3)
    n = 4000
    m = 50 + 0.01 * rng.standard_normal(n).cumsum()
    side = rng.choice([-1.0, 1.0], n)
    price = np.round((m + side * rng.uniform(0, 0.05, n)) * 100) / 100
    lr = lee_ready(price, m - 0.05, m + 0.05)
    assert np.mean(lr == side) > np.mean(tick_test(price) == side)
