r"""
Gatheral's SVI
==============

Jim Gatheral presented the stochastic volatility inspired (SVI)
parametrisation at Global Derivatives in 2004: the total implied
variance :math:`w(k) = \sigma_{BS}^2 T` in log-moneyness :math:`k`,

.. math:: w(k) = a + b\left[\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right],

a hyperbola whose asymptotes are the linear wings Lee's theorem
requires. It is the large-maturity limit of Heston's smile, and five
parameters fit equity smiles remarkably well. Durrleman's function
:math:`g(k)` must stay non-negative for the implied density to be a
density; here SVI is fitted to a Heston smile and checked.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.stochastic_volatility import HestonModel, durrleman_g, svi_fit
from financialkit.stochastic_volatility.visualizers import plot_smile

S, r = 100.0, 0.0
m = HestonModel(0.04, 1.5, 0.04, 0.6, -0.7)
k = np.linspace(-0.6, 0.4, 21)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for T in (0.5, 1.0, 2.0):
    iv = m.implied_vol(S, S * np.exp(k), T, r)
    fit = svi_fit(k, iv**2 * T, T)
    rms = np.sqrt(np.mean((fit.implied_vol(k) - iv) ** 2))
    print(f"T = {T}: a = {fit.a:.4f}, b = {fit.b:.4f}, rho = {fit.rho:.3f}, m = {fit.m:.3f}, sigma = {fit.sigma:.3f}; rms vol error {rms * 1e4:.1f} bp")
    line = ax1.plot(k, iv, "o", ms=3)[0]
    kk = np.linspace(-1.0, 0.8, 200)
    plot_smile(S * np.exp(kk), fit.implied_vol(kk), forward=S, ax=ax1, label=f"SVI, T = {T}", color=line.get_color())
    ax2.plot(kk, durrleman_g(kk, fit.a, fit.b, fit.rho, fit.m, fit.sigma), label=f"T = {T}")
ax2.axhline(0.0, color="k", lw=0.8)
ax2.set_xlabel("log-moneyness k")
ax2.set_ylabel("Durrleman g(k)")
ax2.legend()
fig.tight_layout()

plt.show()
