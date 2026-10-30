r"""
Corwin and Schultz's high-low spread estimator
==============================================

Daily highs are usually buys at the ask and lows sells at the bid, so
the high-low range contains the spread as well as volatility. Shane
Corwin and Paul Schultz (2012) noticed that the volatility part of the
squared range doubles over two days while the spread part does not, and
solved for the spread from two one-day ranges and one two-day range.
It needs only daily data, yet is far more precise than Roll's estimator
from daily closes.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.market_microstructure import corwin_schultz_spread, roll_spread

rng = np.random.default_rng(3)


def daily_bars(days, spread, sigma, rng, m=390):
    # One-minute values; the high trades at the ask, the low at the bid and the close at either.
    value = np.exp(np.cumsum(sigma / np.sqrt(m) * rng.standard_normal(days * m)).reshape(days, m))
    close = value[:, -1] * (1 + 0.5 * spread * rng.choice([-1.0, 1.0], days))
    return value.max(1) * (1 + spread / 2), value.min(1) * (1 - spread / 2), close


# %%
# Monthly estimates
# -----------------
#
# Twelve months of 21 trading days for a stock with a 1% spread and 2%
# daily volatility. Corwin and Schultz set negative two-day estimates to
# zero and average them over the month; that clipping biases the monthly
# figure upward when the spread is small next to volatility, while the
# unclipped average is unbiased but can be negative.

high, low, close = daily_bars(252, 0.01, 0.02, rng)
cs = [corwin_schultz_spread(high[i : i + 21], low[i : i + 21]).mean() for i in range(0, 252, 21)]
roll = [roll_spread(np.log(close[i : i + 21])) for i in range(0, 252, 21)]
print("Corwin-Schultz:", np.round(cs, 4))
print(f"year, unclipped: {corwin_schultz_spread(high, low, clip=False).mean():.4f}")
print("Roll (closes): ", np.round(roll, 4))
assert abs(corwin_schultz_spread(high, low, clip=False).mean() - 0.01) < 0.002

# %%
# Many stocks, true against estimated
# -----------------------------------

true = rng.uniform(0.002, 0.03, 60)
est_cs, est_roll = [], []
for s in true:
    h, l, c = daily_bars(63, s, 0.02, rng)
    est_cs.append(corwin_schultz_spread(h, l).mean())
    est_roll.append(roll_spread(np.log(c)))
rmse_cs, rmse_roll = np.sqrt(np.mean((np.array(est_cs) - true) ** 2)), np.sqrt(np.nanmean((np.array(est_roll) - true) ** 2))
assert rmse_cs < rmse_roll
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(true, est_cs, "o", color="C0", label="Corwin-Schultz")
ax.plot(true, est_roll, "x", color="C1", label=f"Roll ({np.isnan(est_roll).mean():.0%} undefined)")
ax.plot([0, 0.03], [0, 0.03], "k--", lw=1)
ax.set_xlabel("true relative spread")
ax.set_ylabel("quarterly estimate")
ax.legend()
fig.tight_layout()

plt.show()
