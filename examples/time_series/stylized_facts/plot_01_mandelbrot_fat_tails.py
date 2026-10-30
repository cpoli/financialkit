r"""
Mandelbrot: fat tails
=====================

Benoit Mandelbrot (1963) plotted the frequency of large changes in cotton
prices on doubly logarithmic paper and found straight lines: the tail
probability :math:`P(|X| > u)` decays like a power :math:`u^{-\alpha}`,
not like the Gaussian :math:`e^{-u^2/2}`. Five-sigma days, which a normal
distribution makes once-in-several-millennia events, happen every few
years. He proposed the stable (Lévy) laws, with :math:`\alpha \approx 1.7`
for cotton; later work settled on tail indices between 3 and 5 for most
returns, but the log-log plot remains the standard diagnostic.

Here the data are a century of daily returns on the U.S. stock market
(the CRSP value-weighted index, from Kenneth French's data library).
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from financialkit.datasets import load_market_daily
from financialkit.time_series import return_statistics
from financialkit.time_series.visualizers import plot_tail

# %%
# The market against a normal with the same variance
# --------------------------------------------------
# Daily returns from July 1926, standardized, next to as many draws from a
# standard normal.

market = load_market_daily()
r = market.returns
z = (r - r.mean()) / r.std()
normal = np.random.default_rng(1963).standard_normal(r.size)
print(f"{r.size} trading days, {market.dates[0]} to {market.dates[-1]}")

fig, ax = plt.subplots()
plot_tail(z, ax=ax, label="U.S. market, daily")
plot_tail(normal, ax=ax, label="normal")
u = np.logspace(0.3, 1.3, 20)
ax.loglog(u, 0.6 * u**-3, "k--", lw=1, label=r"slope $-3$")
ax.set_ylim(1 / r.size, 1)
ax.set_xlim(0.1, 30)
ax.legend()
ax.set_title("Log-log tails: the market's is a straight line")

# %%
# Moments
# -------
# The normal sample has zero excess kurtosis. The market's is about 16, and
# Jarque-Bera rejects normality at any level.

for name, x in (("U.S. market", r), ("normal", normal)):
    s = return_statistics(x)
    print(f"{name:12s} excess kurtosis {s.excess_kurtosis:6.1f}  JB p-value {s.jarque_bera_p_value:.2g}")
s = return_statistics(r)
assert s.excess_kurtosis > 10 and s.jarque_bera_p_value < 1e-10
assert abs(return_statistics(normal).excess_kurtosis) < 0.2

# %%
# How often are there 5-sigma days?
# ---------------------------------
# About once a year in the market, against once in 7,000 years for a
# normal. The worst day, 19 October 1987, was a 16-sigma fall.

for k in (5, 10):
    freq = np.mean(np.abs(z) > k)
    theory = 1 / (2 * stats.norm.sf(k) * 252)
    print(f"|move| > {k:2d} sd: {np.sum(np.abs(z) > k):3d} days, one every {1 / (freq * 252):5.1f} years; normal theory one every {theory:.2g} years")
print(f"worst day {market.dates[np.argmin(r)]}: {r.min():.1%}, {z.min():.0f} sd")
assert 1 / (np.mean(np.abs(z) > 5) * 252) < 5
assert str(market.dates[np.argmin(r)]) == "1987-10-19"
plt.show()
