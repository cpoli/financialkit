r"""
Fama-French three factors
=========================

The CAPM fails on portfolios sorted by size and book-to-market: small
stocks and value stocks earn more than their market betas justify.
Eugene Fama and Kenneth French (1993) added two long-short factors built
from the same sorts, SMB (small minus big) and HML (high minus low
book-to-market),

.. math::

   r^e_{it} = \alpha_i + b_i\,MKT_t + s_i\,SMB_t + h_i\,HML_t + \varepsilon_{it},

and found that the alphas of 25 size-value portfolios nearly vanish.
Here are their test assets and factors, from French's data library: the
25 value-weighted portfolios and the factors, monthly from July 1963.
French builds SMB and HML from stock-level sorts as
:func:`~financialkit.asset_pricing.sort_portfolios` and
:func:`~financialkit.asset_pricing.long_short_factor` do.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.asset_pricing import grs_test, time_series_regression
from financialkit.asset_pricing.visualizers import plot_alphas
from financialkit.datasets import load_fama_french_25, load_fama_french_factors


def regressions(start: str, end: str):
    """CAPM and three-factor regressions of the 25 portfolios over a window."""
    ports = load_fama_french_25().window(start, end)
    f = load_fama_french_factors().window(start, end)
    assert np.array_equal(ports.dates, f.dates)
    excess = ports.returns - f.risk_free[:, None]
    ff3 = np.column_stack([f.mkt_rf, f.smb, f.hml])
    return time_series_regression(excess, f.mkt_rf), time_series_regression(excess, ff3), grs_test(excess, f.mkt_rf), grs_test(excess, ff3)


# %%
# Fama and French's own sample, 1963-1991
# ---------------------------------------
# GRS rejects the CAPM, but not the three-factor model at 5%.

capm, ff3, g1, g3 = regressions("1963-07", "1991-12")
print(f"CAPM: GRS = {g1.statistic:.2f}, p = {g1.p_value:.3g}")
print(f"FF3:  GRS = {g3.statistic:.2f}, p = {g3.p_value:.3g}")
assert g1.p_value < 0.01 and g3.p_value > 0.05

# %%
# The full sample
# ---------------
# Three more decades of data reject both models: the small-growth corner
# (``me1_bm1``) loses about 6% a year against either. Still, the
# three-factor alphas are on average less than half the CAPM's.

capm, ff3, g1, g3 = regressions("1963-07", "2026-12")
print(f"CAPM: GRS = {g1.statistic:.2f}, p = {g1.p_value:.2g}, mean |alpha| {12 * np.abs(capm.alphas).mean():.2%} a year")
print(f"FF3:  GRS = {g3.statistic:.2f}, p = {g3.p_value:.2g}, mean |alpha| {12 * np.abs(ff3.alphas).mean():.2%} a year")
print(f"small growth alpha: CAPM {12 * capm.alphas[0]:.1%}, FF3 {12 * ff3.alphas[0]:.1%} a year")
assert np.abs(ff3.alphas).mean() < 0.5 * np.abs(capm.alphas).mean()
assert capm.alphas[0] < 0 and ff3.alphas[0] < 0

fig, ax = plt.subplots(figsize=(10, 4))
plot_alphas(capm, ax=ax, label="CAPM", offset=-0.2, width=0.4)
plot_alphas(ff3, ax=ax, label="Fama-French 3", offset=0.2, width=0.4)
ax.set_xlabel("portfolio (size quintile x value quintile)")
fig.tight_layout()

plt.show()
