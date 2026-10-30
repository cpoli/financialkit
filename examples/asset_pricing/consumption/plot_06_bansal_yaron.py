r"""
Bansal-Yaron long-run risk
==========================

Ravi Bansal and Amir Yaron (2004) gave consumption growth a small,
very persistent predictable component :math:`x_t`, too small to detect
in a century of data, plus slowly moving volatility. With Epstein-Zin
preferences, a risk aversion of 10 and an elasticity of intertemporal
substitution :math:`\psi = 1.5`, the investor fears news about
long-run growth much more than a shock to today's consumption.
Dividends load three times on :math:`x_t`, so stocks are very exposed
to that news. The model produces a large equity premium, a low and
stable riskless rate, and volatile prices.

The model is solved by Campbell-Shiller log-linearisation, at Bansal
and Yaron's monthly calibration.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.asset_pricing import BansalYaron, LucasTree

by = BansalYaron()
m = by.moments(n_years=5000)
print(f"annual: riskless {np.log(m.risk_free):.2%}, premium {m.equity_premium:.2%}, volatility {m.equity_vol:.1%}, P/D {m.price_dividend:.1f}")
print(f"log premium from the closed form: {12 * by.equity_premium():.2%}")
lucas = LucasTree(0.998**12, 10.0, 0.018, 0.0078 * np.sqrt(12)).moments()
print(f"Lucas tree, gamma = 10, same consumption volatility: premium {lucas.equity_premium:.2%}")
assert m.equity_premium > 0.04 and m.equity_vol > 0.15
assert m.equity_premium > 5 * lucas.equity_premium

# The simulated monthly log excess return matches the closed form.
path = by.simulate(12 * 20000, rng=1)
excess = np.log(path.equity_return / path.risk_free)
assert abs(excess.mean() - by.equity_premium()) < 4 * excess.std() / np.sqrt(excess.size)

# %%
# Where the premium comes from
# ----------------------------
#
# The market return has no exposure to the short-run shock
# :math:`\eta`, so its whole premium is the price of long-run growth
# news times the market's exposure to it, plus a small volatility term,
# less the Jensen term :math:`\tfrac12 \mathrm{Var}(r_m)`.

s2, sw2 = by.sigma**2, by.sigma_w**2
parts = {
    "long-run growth": 12 * by.lambda_e * by.beta_e * s2,
    "volatility": 12 * by.lambda_w * by.beta_w * sw2,
    "Jensen term": -6 * ((by.beta_e**2 + by.phi_d**2) * s2 + by.beta_w**2 * sw2),
}
for k, v in parts.items():
    print(f"{k:>16}: {v:+.2%}")
assert np.isclose(sum(parts.values()), 12 * by.equity_premium())
assert parts["long-run growth"] > 4 * parts["volatility"]

# %%
# Why the elasticity of substitution must exceed one
# --------------------------------------------------
#
# With :math:`\psi > 1`, good news about growth raises asset prices
# (:math:`A_1 > 0`), so stocks fall exactly when long-run prospects
# worsen. Below one, the income effect dominates. The price-consumption
# ratio then falls on good news, the price of long-run risk shrinks,
# and so does the premium.

psis = np.r_[np.arange(0.3, 0.95, 0.1), np.arange(1.1, 2.05, 0.1)]
models = [BansalYaron(psi=p) for p in psis]
premia = np.array([12 * b.equity_premium() for b in models])
rf = np.array([12 * b.risk_free_rate() for b in models])
assert all(b.A1 < 0 for b in models[:7]) and all(b.A1 > 0 for b in models[7:])
assert premia[7:].min() > premia[:7].max()

fig, ax = plt.subplots(figsize=(7, 4))
lo, hi = psis < 1, psis > 1
for mask in (lo, hi):
    ax.plot(psis[mask], 100 * premia[mask], "o-", color="C0")
    ax.plot(psis[mask], 100 * rf[mask], "s-", color="C1")
ax.plot([], [], "o-", color="C0", label="log equity premium")
ax.plot([], [], "s-", color="C1", label="riskless rate")
ax.axvline(1.0, color="0.6", ls=":")
ax.set_xlabel("elasticity of intertemporal substitution psi")
ax.set_ylabel("% a year")
ax.legend()
fig.tight_layout()

plt.show()
