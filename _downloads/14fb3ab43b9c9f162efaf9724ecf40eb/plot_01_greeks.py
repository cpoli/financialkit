r"""
The Black-Scholes Greeks
========================

The Black-Scholes argument hedges an option with :math:`\Delta =
\partial V/\partial S` shares; the other partial derivatives measure
how that hedge and the option's value respond to everything else. The
Greeks, named by traders after the 1973 paper, satisfy the pricing PDE
itself,

.. math:: \Theta + \tfrac12 \sigma^2 S^2 \Gamma + (r - q) S \Delta = r V,

so time decay is the price of positive gamma. Each analytic Greek is
checked here against a finite difference of the price.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import black_scholes_greeks, black_scholes_price, finite_difference_greeks
from financialkit.options.visualizers import plot_greeks

# %%
# Hull's Example 19.1, analytic and by finite differences
# -------------------------------------------------------

args = (49.0, 50.0, 0.3846, 0.05, 0.20)
g = black_scholes_greeks(*args)
fd = finite_difference_greeks(black_scholes_price, *args)
for name in ("delta", "gamma", "vega", "theta", "rho", "epsilon", "vanna", "volga"):
    print(f"{name:>7}: analytic {getattr(g, name):+10.5f}   finite difference {getattr(fd, name):+10.5f}")

# %%
# The PDE holds across spot
# -------------------------

S = np.linspace(50, 150, 201)
K, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.25
g = black_scholes_greeks(S, K, T, r, sigma, q)
residual = g.theta + 0.5 * sigma**2 * S**2 * g.gamma + r * S * g.delta - r * black_scholes_price(S, K, T, r, sigma)
print(f"max PDE residual: {np.max(np.abs(residual)):.1e}")

# %%
# The Greeks of a call against spot
# ---------------------------------

axes = plot_greeks(S, g)
for ax in axes:
    ax.axvline(K, color="k", lw=0.5)
plt.gcf().tight_layout()

plt.show()
