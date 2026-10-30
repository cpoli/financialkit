r"""
Margrabe exchange options
=========================

William Margrabe (1978) priced the right to exchange one asset for
another, :math:`(S_1 - S_2)^+`. Measured in units of the second asset,
it is a call on the ratio :math:`S_1/S_2` struck at 1, so the interest
rate drops out and only the volatility of the ratio,
:math:`\sigma^2 = \sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2`,
matters. Change of numeraire, the idea behind this one-line proof,
became one of the central tools of derivatives pricing.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.exotic_options import margrabe_price
from financialkit.integrators import correlated_increments

S1, S2, T, s1, s2 = 100.0, 100.0, 1.0, 0.25, 0.3
rho = np.linspace(-0.95, 0.95, 50)

# %%
# Correlation drives the price
# ----------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(rho, margrabe_price(S1, S2, T, s1, s2, rho), label="closed form")
for c in (-0.6, 0.0, 0.6):
    W = correlated_increments([[1.0, c], [c, 1.0]], T, 1, 20_000, rng=int(10 * c) + 10)[:, 0, :]
    A = S1 * np.exp(-0.5 * s1**2 * T + s1 * W[:, 0])
    B = S2 * np.exp(-0.5 * s2**2 * T + s2 * W[:, 1])
    y = np.maximum(A - B, 0.0)  # r drops out: simulate with zero rate
    axes[0].errorbar(c, y.mean(), yerr=2 * y.std() / np.sqrt(y.size), fmt="k.", capsize=3)
    assert abs(y.mean() - margrabe_price(S1, S2, T, s1, s2, c)) < 4 * y.std() / np.sqrt(y.size)
axes[0].set_xlabel(r"correlation $\rho$")
axes[0].set_ylabel("exchange option")
axes[0].legend()

# %%
# Only the ratio's volatility matters
# -----------------------------------
# Pairs of volatilities with the same ratio volatility give the same price.

target = 0.3
for s_1 in (0.2, 0.3, 0.4):
    s_2 = np.linspace(0.05, 0.6, 200)
    c = (s_1**2 + s_2**2 - target**2) / (2 * s_1 * s_2)
    ok = np.abs(c) < 1
    axes[1].plot(s_2[ok], c[ok], label=rf"$\sigma_1$ = {s_1}")
    assert np.allclose(margrabe_price(100.0, 100.0, 1.0, s_1, s_2[ok], c[ok]), margrabe_price(100.0, 100.0, 1.0, 0.3, 0.0, 0.0), rtol=1e-10)
axes[1].set_xlabel(r"$\sigma_2$")
axes[1].set_ylabel(r"$\rho$ giving a ratio volatility of 30%")
axes[1].set_title(f"all priced at {margrabe_price(100.0, 100.0, 1.0, 0.3, 0.0, 0.0):.4f}")
axes[1].legend()
fig.tight_layout()
plt.show()
