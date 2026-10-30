r"""
Campbell-Cochrane habit formation
=================================

John Campbell and John Cochrane (1999) kept Mehra and Prescott's i.i.d.
consumption growth and power utility, but applied the utility to
consumption above a slowly moving habit :math:`X_t`. The surplus ratio
:math:`S_t = (C_t - X_t)/C_t` sets the local curvature
:math:`\gamma / S_t`. In bad times consumption falls toward the habit
and the investor becomes very risk averse, so prices fall and expected
returns rise. The habit's sensitivity :math:`\lambda(s)` is chosen to
keep the riskless rate constant, which avoids the volatile interest
rates of earlier habit models.

The calibration is theirs: :math:`\gamma = 2`, consumption growth 1.89%
with a volatility of 1.50%, surplus persistence 0.87 a year and a
riskless rate of 0.94%. The model is simulated monthly.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.asset_pricing import CampbellCochrane, LucasTree

cc = CampbellCochrane()
print(f"steady-state surplus ratio {cc.S_bar:.3f}, largest {np.exp(cc.s_max):.3f}, annual beta {cc.beta**12:.3f}")

S = np.linspace(0.005, np.exp(cc.s_max), 200)
pc, sharpe = cc.price_consumption(np.log(S)), cc.max_sharpe_ratio(np.log(S))
# Prices are procyclical and the price of risk countercyclical.
assert np.all(np.diff(pc) > 0) and np.all(np.diff(sharpe) < 0)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
ax[0].plot(S, pc)
ax[0].axvline(cc.S_bar, color="0.6", ls=":")
ax[0].set_xlabel("surplus consumption ratio S")
ax[0].set_ylabel("price / annual consumption")
ax[1].plot(S, sharpe, color="C1")
ax[1].axvline(cc.S_bar, color="0.6", ls=":", label="steady state")
ax[1].set_xlabel("surplus consumption ratio S")
ax[1].set_ylabel("maximum Sharpe ratio (annual)")
ax[1].legend()
fig.tight_layout()

# %%
# The equity premium with a risk aversion of two
# ----------------------------------------------
#
# Lucas's tree with the same :math:`\gamma = 2` and the same consumption
# process gives a premium of a few hundredths of a percent. The habit
# economy gives about 6% with a Sharpe ratio near 0.5, and it keeps the
# riskless rate at 0.94%.

m = cc.moments(n_years=5000)
lucas = LucasTree(0.99, 2.0, 0.0189, 0.015).moments()
print(f"habit: riskless {m.risk_free - 1:.2%}, premium {m.equity_premium:.2%}, volatility {m.equity_vol:.1%}")
print(f"       Sharpe ratio {m.sharpe_ratio:.2f}, price / consumption {m.price_dividend:.1f}")
print(f"Lucas tree, gamma = 2: premium {lucas.equity_premium:.3%}")
assert m.equity_premium > 0.05 and lucas.equity_premium < 0.001
assert abs(np.log(m.risk_free) - 0.0094) < 0.002
# The riskless rate is exactly constant below s_max.
rf = np.log(cc.simulate(1200, rng=1).risk_free) * 12
assert np.allclose(rf[cc.surplus_path(1200, rng=1)[:-1] <= cc.s_max], 0.0094)

# %%
# A century of surplus and prices
# -------------------------------
#
# The price-consumption ratio tracks the surplus ratio, so prices fall
# in recessions while the riskless rate does not move.

n = 12 * 100
s = cc.surplus_path(n, rng=2)
years = np.arange(n + 1) / 12
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(years, np.exp(s), label="surplus ratio S")
ax.set_xlabel("years")
ax.set_ylabel("S")
ax2 = ax.twinx()
ax2.plot(years, cc.price_consumption(s), color="C1", label="price / consumption")
ax2.set_ylabel("price / consumption")
fig.legend(loc="upper right")
fig.tight_layout()

plt.show()
