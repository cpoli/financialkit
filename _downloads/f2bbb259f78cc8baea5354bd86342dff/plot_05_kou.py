r"""
Kou's double-exponential jumps
==============================

Steven Kou (2002) took the log jump sizes to be exponential, with rate
:math:`\eta_1` upward (probability :math:`p`) and :math:`\eta_2`
downward. The two exponential tails give the asymmetric, leptokurtic
returns seen in data, and because the exponential distribution is
memoryless the overshoot over a barrier is again exponential, which
makes barrier and lookback options tractable. The tails also cap the
moments, :math:`E[S_T^{s}] < \infty` only for :math:`-\eta_2 < s < \eta_1`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.stochastic_volatility import KouModel, MertonJumpModel, cos_density
from financialkit.stochastic_volatility.visualizers import plot_density

m = KouModel(sigma=0.16, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)
print(f"call S = 100, K = 98, T = 0.5, r = 5%: {m.price(100.0, 98.0, 0.5, 0.05):.5f} (Kou 2002: 9.14732)")

x = m.sample_terminal(0.5, 200_000, rng=0)
pay = np.exp(-0.025) * np.maximum(100 * np.exp(0.025) * np.exp(x) - 98.0, 0.0)
print(f"Monte Carlo: {pay.mean():.4f} +/- {2 * pay.std() / np.sqrt(pay.size):.4f}")

# %%
# Exponential against Gaussian jump tails
# ---------------------------------------
#
# A Merton model with the same jump intensity and the same jump mean and
# variance has Gaussian jump tails; on a log scale Kou's tails are
# straight lines, Merton's parabolas.

p, e1, e2 = m.p, m.eta1, m.eta2
mean = p / e1 - (1 - p) / e2
var = 2 * p / e1**2 + 2 * (1 - p) / e2**2 - mean**2
merton = MertonJumpModel(0.16, 1.0, mean, np.sqrt(var))
grid = np.linspace(-2.5, 1.5, 400)
fig, ax = plt.subplots(figsize=(7, 4))
plot_density(grid, np.maximum(cos_density(m, grid, 1.0, N=1024), 1e-12), ax=ax, label="Kou", log=True)
plot_density(grid, np.maximum(cos_density(merton, grid, 1.0, N=1024), 1e-12), ax=ax, label="Merton, matched jumps", log=True)
ax.set_ylim(1e-8, 5)

plt.show()
