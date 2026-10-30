r"""
Bates: jumps and stochastic volatility
======================================

David Bates (1996), studying Deutsche mark options, found that neither
jumps nor stochastic volatility alone fit the smile at every maturity.
Diffusive stochastic volatility needs time to build up a skew, so
Heston's short-dated smiles are too flat; jumps act at once but their
effect washes out with maturity. Putting Merton's lognormal jumps on
top of Heston gives both: the characteristic function is simply the
product of the two.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.stochastic_volatility import BatesModel, HestonModel, MertonJumpModel
from financialkit.stochastic_volatility.visualizers import plot_smile

S, r = 100.0, 0.02
heston = HestonModel(0.03, 2.0, 0.03, 0.4, -0.6)
bates = BatesModel(0.03, 2.0, 0.03, 0.4, -0.6, lam=0.3, mu_j=-0.12, delta=0.08)

u = np.linspace(-10, 10, 5)
product = np.allclose(bates.char_func(u, 1.0), heston.char_func(u, 1.0) * MertonJumpModel(0.0, 0.3, -0.12, 0.08).char_func(u, 1.0))
print("phi_Bates = phi_Heston * phi_jumps:", product)
assert product

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
extra_skew = []
for ax, T in zip(axes, (0.08, 0.5, 3.0), strict=True):
    F = S * np.exp(r * T)
    K = F * np.exp(np.linspace(-0.3, 0.2, 21) * np.sqrt(max(T, 0.25)))
    iv_heston, iv_bates = heston.implied_vol(S, K, T, r), bates.implied_vol(S, K, T, r)
    extra_skew.append((iv_bates[0] - iv_bates[-1]) - (iv_heston[0] - iv_heston[-1]))
    plot_smile(K, iv_heston, forward=F, ax=ax, label="Heston")
    plot_smile(K, iv_bates, forward=F, ax=ax, label="Bates")
    ax.set_title(f"T = {T}")
fig.tight_layout()
# The jumps steepen the short-dated skew, and their effect has washed out by three years.
assert extra_skew[0] > 0.03 and abs(extra_skew[2]) < 0.01

plt.show()
