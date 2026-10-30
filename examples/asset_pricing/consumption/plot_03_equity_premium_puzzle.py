r"""
The equity premium puzzle
=========================

Rajnish Mehra and Edward Prescott (1985) calibrated a two-state version
of Lucas's economy to US consumption growth from 1889 to 1978 (mean
1.8%, standard deviation 3.6%, autocorrelation -0.14). Over the same
period stocks earned 6.98% a year and short-term bills 0.80%: a premium
of 6.18%. Searching over every time preference :math:`\beta < 1` and
risk aversion :math:`\gamma \le 10`, they could produce a premium of at
most 0.35% while keeping the riskless rate below 4%. Consumption is
simply too smooth for stocks to be that risky to a power-utility agent.

The premium here is measured, not assumed: the CRSP market and one-month
T-bill returns from Kenneth French's data library, compounded each
calendar year from 1927 and deflated by Shiller's CPI. The consumption
moments are still Mehra and Prescott's; financialkit bundles no
consumption series.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.asset_pricing import MehraPrescott, implied_risk_aversion
from financialkit.datasets import load_fama_french_factors, load_shiller

# %%
# The premium in the data
# -----------------------

factors, shiller = load_fama_french_factors(), load_shiller()
year_of = factors.dates.astype("datetime64[Y]").astype(int) + 1970
years = np.arange(1927, 2026)
stocks = np.array([np.prod(1 + (factors.mkt_rf + factors.risk_free)[year_of == y]) - 1 for y in years])
bills = np.array([np.prod(1 + factors.risk_free[year_of == y]) - 1 for y in years])
december = shiller.dates.astype(int) % 12 == 11
cpi = dict(zip(shiller.dates[december].astype("datetime64[Y]").astype(int) + 1970, shiller.cpi[december], strict=True))
inflation = np.array([cpi[y] / cpi[y - 1] - 1 for y in years])
real_stocks, real_bills = (1 + stocks) / (1 + inflation) - 1, (1 + bills) / (1 + inflation) - 1
premium, vol, r_f = float(np.mean(real_stocks - real_bills)), float(np.std(real_stocks, ddof=1)), float(np.mean(real_bills))
print(f"{years[0]}-{years[-1]}: real stocks {real_stocks.mean():.2%} (sd {vol:.1%}), real bills {r_f:.2%}, premium {premium:.2%}")
assert 0.06 < premium < 0.10 and abs(r_f) < 0.02

# %%
# What the model can produce
# --------------------------

points = []
for g in np.linspace(0.1, 10, 40):
    for b in np.linspace(0.85, 0.9999, 40):
        try:
            m = MehraPrescott(b, g).moments()
        except ValueError:
            continue
        points.append((m.risk_free - 1, m.equity_premium))
pts = np.array(points)
ok = (pts[:, 0] >= 0) & (pts[:, 0] <= 0.04)
print(f"largest premium with 0 <= r_f <= 4%: {pts[ok, 1].max():.2%}")
# Mehra and Prescott's 0.35%, a small fraction of the measured premium.
assert pts[ok, 1].max() < 0.004 < premium / 10

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(100 * pts[:, 0], 100 * pts[:, 1], s=4, c="0.6", label="model, gamma <= 10, beta < 1")
ax.scatter(100 * pts[ok, 0], 100 * pts[ok, 1], s=4, c="C0", label="... with 0 <= r_f <= 4%")
ax.plot(100 * r_f, 100 * premium, "*", color="C3", ms=14, label=f"US data {years[0]}-{years[-1]}")
ax.plot(0.80, 6.18, "*", color="C1", ms=10, label="Mehra-Prescott, 1889-1978")
ax.set_xlim(-2, 20)
ax.set_xlabel("average riskless rate (%)")
ax.set_ylabel("equity premium (%)")
ax.legend()
fig.tight_layout()

# %%
# How much risk aversion would it take?
# -------------------------------------
#
# With the measured premium and equity volatility and the 3.6%
# consumption volatility, the consumption CAPM needs :math:`\gamma` above
# 10 even if stocks and consumption were perfectly correlated, and about
# 30 with a realistic correlation of 0.4.

for rho in (1.0, 0.4):
    print(f"correlation {rho}: gamma = {implied_risk_aversion(premium, vol, 0.036, rho):.0f}")
assert implied_risk_aversion(premium, vol, 0.036, 1.0) > 10
assert implied_risk_aversion(premium, vol, 0.036, 0.4) > 20

gammas = np.linspace(1, 40, 40)
prem = [MehraPrescott(0.99, g).moments().equity_premium for g in gammas]
assert gammas[np.argmax(np.array(prem) >= premium)] > 10
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(gammas, 100 * np.array(prem))
ax.axhline(100 * premium, color="C3", ls="--", label="measured premium")
ax.set_xlabel("risk aversion gamma (beta = 0.99)")
ax.set_ylabel("model premium (%)")
ax.legend()
fig.tight_layout()

plt.show()
