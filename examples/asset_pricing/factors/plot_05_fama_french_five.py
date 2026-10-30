r"""
Fama-French five factors
========================

Rewriting the dividend discount model as
:math:`M/B = \sum_t E[Y_t - dB_t]/(1 + r)^t / B` shows that, holding
price-to-book fixed, firms with higher profitability :math:`Y` or lower
investment :math:`dB` must have higher expected returns. Eugene Fama and
Kenneth French (2015) added two factors built from those
characteristics to their three. RMW is robust minus weak operating
profitability, and CMA is conservative minus aggressive investment.
Each comes from a 2 x 3 sort on size and the characteristic. The
five-factor model priced the profitability and investment anomalies,
and HML became redundant: its average return was explained by its
loadings on the other four.

The cross-section is synthetic: 400 stocks over 40 years. Value firms
invest less, and the latent value factor carries no premium of its own,
so all of HML's premium comes through investment.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.asset_pricing import fama_french_five, grs_test, sort_portfolios, time_series_regression

rng = np.random.default_rng(0)
T, N = 480, 400
size, prof, inv = rng.standard_normal((3, N))
bm = -0.7 * inv + np.sqrt(1 - 0.49) * rng.standard_normal(N)
# latent factors: market, size, value, profitability, investment (value has no premium)
means = np.array([0.006, 0.002, 0.0, 0.005, 0.005])
vols = np.array([0.045, 0.03, 0.015, 0.015, 0.015])
f = means + vols * rng.standard_normal((T, 5))
B = np.column_stack([1 + 0.2 * rng.standard_normal(N), -0.5 * size, 0.5 * bm, 0.5 * prof, -0.5 * inv])
returns = f @ B.T + 0.06 * rng.standard_normal((T, N))

factors = fama_french_five(returns, returns.mean(axis=1), size, bm, prof, inv)
names = ["MKT", "SMB", "HML", "RMW", "CMA"]
for n, x in zip(names, factors.T, strict=True):
    print(f"{n}: mean {x.mean():.2%} a month, t = {x.mean() / x.std() * np.sqrt(T):.1f}")

# %%
# Pricing profitability and investment portfolios
# -----------------------------------------------
#
# Deciles sorted on profitability and on investment are the anomalies
# that the three-factor model misses. The GRS test rejects the three
# factors and does not reject the five.

test = np.hstack([sort_portfolios(prof, returns, 10), sort_portfolios(-inv, returns, 10)])
ff3 = time_series_regression(test, factors[:, :3])
ff5 = time_series_regression(test, factors)
grs3, grs5 = grs_test(test, factors[:, :3]), grs_test(test, factors)
print(f"GRS p-value: three factors {grs3.p_value:.1e}, five factors {grs5.p_value:.2f}")
assert grs3.reject() and not grs5.reject()
assert np.abs(ff5.alphas).mean() < 0.5 * np.abs(ff3.alphas).mean()

fig, ax = plt.subplots(figsize=(8, 3.8))
k = np.arange(20)
ax.bar(k - 0.2, 100 * ff3.alphas, 0.4, label="three factors")
ax.bar(k + 0.2, 100 * ff5.alphas, 0.4, label="five factors")
ax.axvline(9.5, color="0.6", ls=":")
ax.set_xticks([4.5, 14.5], ["profitability deciles (weak to robust)", "investment deciles (aggressive to conservative)"])
ax.set_ylabel("alpha (% a month)")
ax.legend()
fig.tight_layout()

# %%
# HML is redundant
# ----------------
#
# Regressing HML on the other four factors leaves no alpha: its premium
# is its negative exposure to investment. This is Fama and French's
# 2015 finding.

span = time_series_regression(factors[:, 2], factors[:, [0, 1, 3, 4]])
print(f"HML on the other four: alpha {span.alphas[0]:.3%} (t = {span.alpha_t[0]:.2f}), CMA loading {span.betas[0, 3]:.2f}")
assert factors[:, 2].mean() / factors[:, 2].std() * np.sqrt(T) > 2
assert abs(span.alpha_t[0]) < 2 and span.betas[0, 3] > 0.3

plt.show()
