"""Tests for the bundled-dataset loaders: shapes, dates, units and known values."""

import numpy as np
import pytest

from financialkit.datasets import (
    load_fama_french_25,
    load_fama_french_factors,
    load_market_daily,
    load_shiller,
    load_treasury_yields,
)


def _monthly_and_increasing(dates):
    assert dates.dtype == np.dtype("datetime64[M]")
    assert np.all(np.diff(dates).astype(int) == 1)


def test_shiller():
    d = load_shiller()
    _monthly_and_increasing(d.dates)
    assert str(d.dates[0]) == "1871-01"
    assert d.price[0] == pytest.approx(4.44) and d.dividend[0] == pytest.approx(0.26)
    assert d.long_rate[0] == pytest.approx(0.0532)  # stored in percent, returned as a decimal
    assert not np.isnan(d.price).any() and not np.isnan(d.cpi).any()
    assert np.isnan(d.dividend[-1])  # the latest months have no dividend yet
    assert d.real_price[-1] == pytest.approx(d.price[-1])
    jan = d.annual()
    assert np.all(jan.dates.astype(int) % 12 == 0) and np.all(np.diff(jan.dates.astype("datetime64[Y]")).astype(int) == 1)
    assert str(d.annual(12).dates[0]) == "1871-12"


def test_market_daily():
    m = load_market_daily()
    assert m.dates.dtype == np.dtype("datetime64[D]") and np.all(np.diff(m.dates).astype(int) > 0)
    assert str(m.dates[0]) == "1926-07-01" and m.excess[0] == pytest.approx(0.0009)
    assert np.isfinite(m.returns).all()
    assert str(m.dates[np.argmin(m.returns)]) == "1987-10-19"
    assert 0 <= m.risk_free.min() and m.risk_free.max() < 0.001


def test_fama_french_factors():
    f = load_fama_french_factors()
    _monthly_and_increasing(f.dates)
    assert str(f.dates[0]) == "1926-07" and f.mkt_rf[0] == pytest.approx(0.0289) and f.risk_free[0] == pytest.approx(0.0022)
    assert str(f.dates[~np.isnan(f.rmw)][0]) == "1963-07" and str(f.dates[~np.isnan(f.cma)][0]) == "1963-07"
    assert str(f.dates[~np.isnan(f.mom)][0]) == "1927-01"
    for x in (f.mkt_rf, f.smb, f.hml, f.risk_free):
        assert np.isfinite(x).all()
    # The equity premium: about 8% a year since 1926.
    assert 0.06 < 12 * f.mkt_rf.mean() < 0.10
    w = f.window("1963-07", "1991-12")
    assert str(w.dates[0]) == "1963-07" and str(w.dates[-1]) == "1991-12" and w.dates.size == 342
    assert np.isfinite(w.rmw).all()


def test_fama_french_25():
    p = load_fama_french_25()
    _monthly_and_increasing(p.dates)
    assert str(p.dates[0]) == "1963-07"
    assert p.returns.shape == (p.dates.size, 25) and np.isfinite(p.returns).all()
    assert p.names[0] == "me1_bm1" and p.names[24] == "me5_bm5" and len(set(p.names)) == 25
    # Monthly returns in decimals: no month moves a portfolio by 100%.
    assert np.abs(p.returns).max() < 1
    f = load_fama_french_factors().window(str(p.dates[0]), str(p.dates[-1]))
    np.testing.assert_array_equal(f.dates, p.dates)
    assert p.window("2000-01").dates[0] == np.datetime64("2000-01")


def test_treasury_yields():
    c = load_treasury_yields()
    _monthly_and_increasing(c.dates)
    assert str(c.dates[0]) == "1981-09"
    np.testing.assert_allclose(c.tenors, [0.25, 0.5, 1, 2, 3, 5, 7, 10])
    assert c.yields.shape == (c.dates.size, 8) and np.isfinite(c.yields).all()
    assert 0 <= c.yields.min() and c.yields.max() < 0.2  # decimals, peaking near 16% in 1981
    assert c.yields[0, -1] == pytest.approx(0.1584)


def test_results_are_frozen():
    d = load_treasury_yields()
    with pytest.raises(AttributeError):
        d.tenors = np.zeros(8)


def test_shiller_real_dividend_is_in_latest_dollars():
    d = load_shiller()
    assert d.real_dividend[0] == pytest.approx(d.dividend[0] * d.cpi[-1] / d.cpi[0])
