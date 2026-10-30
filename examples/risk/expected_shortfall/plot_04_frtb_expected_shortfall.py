r"""
FRTB: 97.5% expected shortfall with liquidity horizons
======================================================

The Basel Committee's Fundamental Review of the Trading Book (2016)
made two changes to internal-model capital. The 99% VaR became a 97.5%
expected shortfall, the same figure for normal returns but much larger
when the tail is fat. And the single ten-day horizon became five
liquidity horizons, 10 to 120 days, one for each risk factor. Ten-day
ES figures are computed with the shocks of ever fewer, ever less liquid
factors, and cascaded:

.. math::

   \mathrm{ES} = \sqrt{\mathrm{ES}_T(P)^2 + \sum_{j \ge 2}
   \left(\mathrm{ES}_T(P, j)\sqrt{\frac{LH_j - LH_{j-1}}{T}}\right)^2}, \qquad T = 10 .
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.risk import expected_shortfall, frtb_expected_shortfall, historical_var
from financialkit.risk.visualizers import plot_frtb_cascade

# %%
# From 99% VaR to 97.5% ES
# ------------------------
#
# The Committee chose 97.5% so that the change would be neutral for
# normal returns: the two figures differ by half a percent. Under
# Student-t returns with four degrees of freedom the ES is clearly
# higher, which is the point. Expected shortfall charges for the shape
# of the tail beyond the quantile.

rng = np.random.default_rng(0)
n = 200_000
normal = rng.standard_normal(n)
t4 = rng.standard_t(4, n) / np.sqrt(2.0)  # unit variance
for name, x in [("normal", normal), ("Student-t(4)", t4)]:
    v, e = historical_var(x, 0.99).var, expected_shortfall(x, 0.975)
    print(f"{name:13s}: 99% VaR {v:.3f}, 97.5% ES {e:.3f}, ratio {e / v:.3f}")
assert abs(expected_shortfall(normal, 0.975) / historical_var(normal, 0.99).var - 1.005) < 0.02
assert expected_shortfall(t4, 0.975) / historical_var(t4, 0.99).var > 1.03

# %%
# A trading book across five liquidity horizons
# ---------------------------------------------
#
# The book's risk factors carry the liquidity horizons that Basel (2016),
# Table 2, gives their categories. Ten-day moves are drawn from a
# multivariate Student-t law (4 degrees of freedom, correlation 0.3),
# standing in for a calibration to a stressed period. The exposures, in
# $ millions per unit move, are illustrative.

factors = ["equity large cap", "equity vol", "IG credit spread", "HY credit spread", "commodity vol"]
horizons = np.array([10, 20, 40, 60, 120])
exposure = np.array([4.0, 1.5, 3.0, 2.0, 1.0])
d, nu = len(factors), 4
corr = np.full((d, d), 0.3) + 0.7 * np.eye(d)
g = rng.standard_normal((50_000, d)) @ np.linalg.cholesky(corr).T
scenarios = 0.02 * g / np.sqrt(rng.chisquare(nu, (50_000, 1)) / nu)


def pnl(dX):
    return -(dX @ exposure)  # long risk: losses when the factors rise


res = frtb_expected_shortfall(pnl, scenarios, horizons)
ten_day = res.es_by_horizon[0]
print(f"ten-day 97.5% ES          {ten_day:6.3f}")
print(f"liquidity-adjusted ES     {res.es:6.3f}  ({res.es / ten_day:.2f}x)")
print(f"if everything took 120 d  {ten_day * np.sqrt(12):6.3f}")
assert ten_day < res.es < ten_day * np.sqrt(12)
assert np.all(np.diff(res.es_by_horizon) < 0)
plot_frtb_cascade(res)

# %%
# Moving one factor to a longer horizon
# -------------------------------------
#
# Reclassifying the equity exposure from large cap (10 days) to small
# cap (20 days) raises capital even though its ten-day risk is unchanged.

for lh in (10, 20, 40, 60, 120):
    h = horizons.copy()
    h[0] = lh
    print(f"equity horizon {lh:3d} days: ES {frtb_expected_shortfall(pnl, scenarios, h).es:.3f}")

plt.show()
