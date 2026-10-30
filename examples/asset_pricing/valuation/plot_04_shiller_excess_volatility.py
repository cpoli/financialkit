r"""
Shiller's excess volatility
===========================

If a price is the rational forecast of discounted future dividends,
:math:`p_t = E_t[p^*_t]`, then the ex-post rational price :math:`p^*_t`
computed from the dividends actually paid equals the price plus an
unforecastable error. An optimal forecast varies less than the thing it
forecasts, so :math:`\sigma(p) \le \sigma(p^*)`. Robert Shiller (1981)
found US stock prices five to thirteen times too volatile for this
bound.

Here is his test on his own data: the real S&P composite price and
dividends each January from 1871 to 1979, from Shiller's
``ie_data.xls``.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.asset_pricing import variance_bound
from financialkit.asset_pricing.visualizers import plot_excess_volatility
from financialkit.datasets import load_shiller

# %%
# Detrending
# ----------
# Like Shiller, divide both series by the exponential growth trend of the
# real price, and discount at the average detrended dividend yield. The
# terminal :math:`p^*` is the sample-average detrended price.

jan = load_shiller().annual()
keep = jan.dates <= np.datetime64("1979-01")
dates, P, D = jan.dates[keep], jan.real_price[keep], jan.real_dividend[keep]
t = np.arange(P.size)
slope, intercept = np.polyfit(t, np.log(P), 1)
trend = np.exp(intercept + slope * t)
p, d = P / trend, D / trend
r = float(np.mean(d / p))
print(f"{dates[0]} to {dates[-1]}: real growth {np.expm1(slope):.2%} a year, discount rate {r:.2%}")

result = variance_bound(p, d, r)
print(f"sd(p) = {result.std_price:.3f}, sd(p*) = {result.std_rational:.3f}, ratio {result.ratio:.1f}")
assert result.violated and result.ratio > 3

# %%
# The plot that started behavioural finance
# -----------------------------------------
# :math:`p^*` is nearly a smooth trend line, because dividends are smooth;
# the price swings far around it.

fig, ax = plt.subplots(figsize=(8, 4))
plot_excess_volatility(result, ax=ax, label="detrended real S&P price")
ax.set_xticks(t[::20], [str(x)[:4] for x in dates[::20]])
ax.set_xlabel("January of")
ax.set_title("Shiller (1981): prices move too much")
fig.tight_layout()

plt.show()
