r"""
Bangia-Diebold-Schuermann-Stroughair: liquidity-adjusted VaR
============================================================

A market VaR values the position at the mid price. Selling it means
crossing half the bid-ask spread, and spreads widen exactly when prices
fall. Bangia, Diebold, Schuermann and Stroughair (1999) added the cost
of liquidating at a stressed relative spread to a lognormal VaR:

.. math:: \mathrm{LVaR} = P\big(1 - e^{-z_\alpha \sigma}\big) + \tfrac12 P(\bar S + a\,\sigma_S).

In their emerging-market currencies the second term was a large share
of the total. Ignoring it understated risk by as much.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.risk import exogenous_liquidity_cost, kupiec_test, liquidity_adjusted_var, var_exceptions

# %%
# A market where spreads widen in a sell-off
# ------------------------------------------
#
# Daily log returns are normal with 1.2% volatility. The relative spread
# is lognormal around 80 bp and correlated -0.6 with the return. Each
# day's realised loss from liquidating a $1 million position is the
# price move plus half that day's spread.

rng = np.random.default_rng(3)
n, sigma, P = 3000, 0.012, 1e6
z = rng.standard_normal((n, 2))
r = sigma * z[:, 0]
spread = np.exp(np.log(0.008) + 0.35 * (-0.6 * z[:, 0] + 0.8 * z[:, 1]))
liquidation_pnl = P * np.expm1(r) - 0.5 * P * spread

# %%
# Estimate on the first three years, backtest on the rest
# -------------------------------------------------------
#
# The market VaR ignores the spread. The liquidity-adjusted VaR adds
# half the mean spread plus :math:`z_\alpha` spread standard deviations.

est = 750
s_hat, S_bar, S_sd = r[:est].std(ddof=1), spread[:est].mean(), spread[:est].std(ddof=1)
lvar = liquidity_adjusted_var(P, s_hat, S_bar, S_sd, 0.99)
market = liquidity_adjusted_var(P, s_hat, 0.0, 0.0, 0.99)
print(f"mean spread {1e4 * S_bar:.0f} bp, spread sd {1e4 * S_sd:.0f} bp")
print(f"market VaR {market:,.0f}, liquidity cost {lvar - market:,.0f}, LVaR {lvar:,.0f} ({(lvar - market) / lvar:.0%} liquidity)")
p_values = {}
for name, v in [("market VaR", market), ("LVaR", lvar)]:
    res = kupiec_test(var_exceptions(liquidation_pnl[est:], v))
    p_values[name] = res.p_value
    print(f"{name:10s}: {res.n_exceptions} exceptions (expected {res.expected:.1f}), Kupiec p = {res.p_value:.3g}")
assert p_values["market VaR"] < 1e-6 < 0.05 < p_values["LVaR"]

fig, ax = plt.subplots(figsize=(8, 3.6))
days = np.arange(est, n)
ax.plot(days, liquidation_pnl[est:], color="0.7", lw=0.5)
ax.axhline(-market, color="C3", label="market VaR")
ax.axhline(-lvar, color="C0", label="liquidity-adjusted VaR")
ax.set_xlabel("day")
ax.set_ylabel("liquidation P&L ($)")
ax.legend(loc="lower left")
fig.tight_layout()

# %%
# How much of the risk is liquidity?
# ----------------------------------
#
# For the same 1.2% volatility, the share of the LVaR due to the spread
# grows with the spread's level and variability. The spreads below are
# illustrative, from a major currency to an illiquid bond.

means = np.array([2, 10, 30, 80, 200]) * 1e-4
sds = 0.5 * means
labels = ["2 bp", "10 bp", "30 bp", "80 bp", "200 bp"]
cost = exogenous_liquidity_cost(P, means, sds, 2.326)
total = liquidity_adjusted_var(P, sigma, means, sds)
share = cost / total
assert np.all(np.diff(share) > 0)
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar(labels, 100 * share)
ax.set_xlabel("mean relative spread (sd = half the mean)")
ax.set_ylabel("liquidity share of LVaR (%)")
fig.tight_layout()

plt.show()
