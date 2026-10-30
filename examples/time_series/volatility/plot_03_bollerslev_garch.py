r"""
Bollerslev's GARCH
==================

Pure ARCH models needed many lags to capture how slowly volatility decays.
Tim Bollerslev (1986), Engle's student, added the lagged variance itself,

.. math:: \sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2,

an ARCH(:math:`\infty`) with geometrically declining weights in three
parameters. Fitted to daily returns almost anywhere, GARCH(1,1) gives
:math:`\alpha \approx 0.05\text{-}0.15` and :math:`\alpha + \beta` just
below one: shocks to volatility are small but long-lived. Variance
forecasts revert to :math:`\bar\sigma^2 = \omega/(1-\alpha-\beta)` at rate
:math:`(\alpha+\beta)^h`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.constants import TRADING_DAYS_PER_YEAR
from financialkit.datasets import load_market_daily
from financialkit.time_series import GARCH
from financialkit.time_series.visualizers import plot_conditional_volatility

# %%
# Fitting GARCH(1,1) by maximum likelihood
# ----------------------------------------
# Daily log returns on the U.S. stock market (CRSP value-weighted, from
# French's data library) since 2000, through the dot-com bust, 2008 and
# March 2020.

ann = TRADING_DAYS_PER_YEAR
market = load_market_daily()
since = market.dates >= np.datetime64("2000-01-01")
dates, r = market.dates[since], np.log1p(market.returns[since])
fit = GARCH.fit(r)
for k, v in fit.params.items():
    print(f"{k:5s} {v:.3g} +/- {fit.std_errors[k]:.2g}")
print(f"persistence {fit.model.persistence:.3f}, half-life {fit.model.half_life:.0f} days")
print(f"long-run volatility {np.sqrt(ann * fit.model.unconditional_variance):.1%} a year")
assert 0.05 < fit.params["alpha"] < 0.2
assert 0.95 < fit.model.persistence < 1

fig, ax = plt.subplots(figsize=(8, 3.5))
plot_conditional_volatility(r, fit.variance, ax=ax)
ax.set_title("Fitted conditional volatility, U.S. market since 2000")
peak = int(np.argmax(fit.variance))
print(f"peak fitted volatility on {dates[peak]}: {np.sqrt(ann * fit.variance[peak]):.0%} a year")
assert str(dates[peak])[:4] in ("2008", "2020")

# %%
# The term structure of volatility forecasts
# ------------------------------------------
# From the calmest and from the most turbulent day, forecasts converge to
# the long-run volatility at the rate set by the persistence.

h = 500
calm = int(np.argmin(fit.variance[500:])) + 500
fig, ax = plt.subplots()
for name, t in (("after the calmest day", calm), ("after the most turbulent day", peak)):
    f = fit.model.forecast(r[: t + 1], h)
    assert np.all(np.diff(f) > 0) if t == calm else np.all(np.diff(f) < 0)
    assert abs(f[-1] / fit.model.unconditional_variance - 1) < 0.05  # converged after two years
    ax.plot(np.arange(1, h + 1), 100 * np.sqrt(ann * f), label=f"{name} ({dates[t]})")
ax.axhline(100 * np.sqrt(ann * fit.model.unconditional_variance), color="k", ls="--", lw=1, label="long-run")
ax.set_xlabel("horizon (days)")
ax.set_ylabel("forecast volatility (% a year)")
ax.legend()
plt.show()
