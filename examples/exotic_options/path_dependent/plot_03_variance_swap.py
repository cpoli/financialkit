r"""
Demeterfi-Derman-Kamal-Zou variance swaps
=========================================

A variance swap pays the realized variance of the returns minus a fixed
strike. Kresimir Demeterfi, Emanuel Derman, Michael Kamal and Joseph Zou
(1999) showed how to replicate it, whatever the volatility process, with
a static strip of out-of-the-money options weighted by :math:`1/K^2`
plus a delta hedge, which gives the fair strike from today's option
prices alone:

.. math:: K_{\mathrm{var}} = \frac{2e^{rT}}{T}\left[\int_0^{F}\frac{P(K)}{K^2}dK + \int_F^\infty \frac{C(K)}{K^2}dK\right].

The same formula, discretised, is the VIX. With a downward skew the
cheap-looking puts carry high implied volatility, and the fair variance
lies above the at-the-money volatility squared.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.exotic_options import annualized_realized_variance, plot_variance_strip, variance_swap_strike
from financialkit.integrators import gbm_paths
from financialkit.options import black_scholes_price

S, T, r = 100.0, 1.0, 0.03
F = S * np.exp(r * T)
K = np.arange(30.0, 300.0, 2.5)


def skew(k):
    return np.clip(0.2 - 0.15 * np.log(k / F), 0.08, None)


def strip(vol):
    c, p = (black_scholes_price(S, K, T, r, vol, option_type=t) for t in ("call", "put"))
    return variance_swap_strike(K, c, p, S, T, r)


flat, skewed = strip(0.2), strip(skew(K))
print(f"flat 20% smile: fair volatility {flat.vol_strike:.4f}")
print(f"skewed smile (ATM 20%): fair volatility {skewed.vol_strike:.4f}")
# The strip recovers a flat 20% to within the strike grid's truncation
# error, and the skew lifts the fair volatility above the at-the-money one.
assert abs(flat.vol_strike - 0.2) < 1e-3
assert skewed.vol_strike > 0.2

# %%
# The replicating strip under the skew
# ------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(K, skew(K))
axes[0].axvline(F, color="k", ls="--", lw=1)
axes[0].set_xlabel("strike K")
axes[0].set_ylabel("implied volatility")
plot_variance_strip(skewed, ax=axes[1])
axes[1].set_xlim(40, 200)

# %%
# Under Black-Scholes, realized variance averages to the fair strike
# ------------------------------------------------------------------

_, paths = gbm_paths(S, r, 0.2, T, 252, 5000, rng=0)
rv = annualized_realized_variance(paths, T)
assert abs(rv.mean() - flat.fair_variance) < 4 * rv.std() / np.sqrt(rv.size)
axes[2].hist(rv, bins=50, density=True, alpha=0.7)
axes[2].axvline(flat.fair_variance, color="k", lw=2, label=f"fair strike {flat.fair_variance:.4f}")
axes[2].axvline(rv.mean(), color="C3", ls="--", label=f"mean realized {rv.mean():.4f}")
axes[2].set_xlabel("realized variance (daily returns)")
axes[2].legend()
fig.tight_layout()
plt.show()
