r"""
Vasicek's mean-reverting short rate
===================================

Oldrich Vasicek (1977) derived the term structure from a short rate
pulled back towards a long-run mean,

.. math:: dr = \kappa(\theta - r)\,dt + \sigma\,dW,

an Ornstein-Uhlenbeck process, together with the market price of risk
that makes bond prices consistent. Bonds are exponential-affine in the
rate, :math:`P = A(\tau) e^{-B(\tau) r}`, and the curve is upward
sloping, humped or inverted depending on where :math:`r_0` stands
against the long yield :math:`\theta - \sigma^2/(2\kappa^2)`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.interest_rate_models import VasicekModel, monte_carlo_bond_prices
from financialkit.interest_rate_models.visualizers import plot_model_yields, plot_short_rate_paths

# %%
# Three curve shapes from one model
# ---------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for r0, shape in ((0.01, "normal"), (0.0555, "humped"), (0.09, "inverted")):
    m = VasicekModel(r0, 0.25, 0.06, 0.02)
    plot_model_yields(m, t_max=30, ax=axes[0], label=f"r0 = {100 * r0:.2f}% ({shape})")
    dy = np.diff(m.yields(np.linspace(0.1, 30, 300)))
    assert {"normal": np.all(dy > 0), "inverted": np.all(dy < 0), "humped": dy[0] > 0 > dy[-1]}[shape]
axes[0].axhline(100 * m.long_yield, color="k", ls=":", lw=1)
axes[0].set_title("Vasicek yield curves")

# %%
# Exact simulation and mean reversion
# -----------------------------------

model = VasicekModel(0.01, 0.25, 0.06, 0.02)
times, r = model.simulate(20.0, 200, 5000, rng=0)
plot_short_rate_paths(times, r, n_show=12, ax=axes[1], label="simulated")
axes[1].plot(times, 100 * (0.06 + (0.01 - 0.06) * np.exp(-0.25 * times)), "w--", lw=1, label="E[r(t)]")
axes[1].legend()

# %%
# Closed form against Monte Carlo
# -------------------------------

res = monte_carlo_bond_prices(model, np.arange(1.0, 21.0), n_steps=200, n_paths=8000, rng=1)
assert np.abs(res.z_scores).max() < 3
axes[2].errorbar(res.maturities, res.z_scores, yerr=1, fmt="o", ms=3)
axes[2].axhspan(-2, 2, color="0.9")
axes[2].set_xlabel("maturity (years)")
axes[2].set_ylabel("(MC - exact) / SE")
axes[2].set_title(f"max |z| = {np.abs(res.z_scores).max():.2f}")
fig.tight_layout()

plt.show()
