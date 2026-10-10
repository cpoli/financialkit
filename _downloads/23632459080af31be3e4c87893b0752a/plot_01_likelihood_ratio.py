r"""
Glynn: likelihood-ratio Greeks
==============================

Bumping a parameter and re-simulating gives a biased, noisy derivative.
Peter Glynn (1987), and Reiman and Weiss (1989), differentiated the
*density* instead of the payoff:

.. math:: \frac{\partial}{\partial\theta} E[f(S_T)] = E\left[f(S_T)\,\frac{\partial \ln p_\theta(S_T)}{\partial\theta}\right].

Under Black-Scholes the delta weight is :math:`Z / (S\sigma\sqrt T)`. The
payoff is never differentiated, so the method prices the Greeks of a
digital, whose payoff has no useful derivative at all.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from financialkit.monte_carlo import likelihood_ratio_greeks
from financialkit.options import black_scholes_greeks

S, T, r, sigma = 100.0, 0.5, 0.03, 0.25
spots = np.linspace(70, 130, 13)

# %%
# Vanilla call: delta, vega and gamma
# -----------------------------------

lr = [likelihood_ratio_greeks(s, 100.0, T, r, sigma, n_paths=50_000, rng=i) for i, s in enumerate(spots)]
bs = black_scholes_greeks(spots, 100.0, T, r, sigma)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, name in zip(axes[:3], ["delta", "vega", "gamma"], strict=True):
    ax.plot(spots, getattr(bs, name), "k", label="Black-Scholes")
    ax.errorbar(spots, [getattr(g, name) for g in lr], yerr=[2 * getattr(g, name + "_se") for g in lr], fmt="o", ms=3, capsize=2, label="likelihood ratio")
    ax.set_xlabel("spot S")
    ax.set_title(name)
axes[0].legend()

# %%
# Digital call delta
# ------------------
# Cash-or-nothing call: :math:`V = e^{-rT}N(d_2)`, delta
# :math:`e^{-rT} n(d_2) / (S\sigma\sqrt T)`.

d2 = (np.log(spots / 100.0) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
dig = [likelihood_ratio_greeks(s, 100.0, T, r, sigma, digital=True, n_paths=50_000, rng=100 + i) for i, s in enumerate(spots)]
axes[3].plot(spots, np.exp(-r * T) * norm.pdf(d2) / (spots * sigma * np.sqrt(T)), "k", label="closed form")
axes[3].errorbar(spots, [g.delta for g in dig], yerr=[2 * g.delta_se for g in dig], fmt="o", ms=3, capsize=2, label="likelihood ratio")
axes[3].set_xlabel("spot S")
axes[3].set_title("digital call delta")
axes[3].legend()
g = lr[6]
print(f"ATM: delta {g.delta:.4f} +/- {g.delta_se:.4f} (BS {bs.delta[6]:.4f}), gamma {g.gamma:.5f} +/- {g.gamma_se:.5f} (BS {bs.gamma[6]:.5f})")
fig.tight_layout()
plt.show()
