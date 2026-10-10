r"""
Black-Litterman: equilibrium plus views
=======================================

Fischer Black and Robert Litterman (1992), at Goldman Sachs, started
mean-variance optimization from the expected returns that make the
market portfolio optimal, :math:`\Pi = \delta\Sigma w_{mkt}`, rather
than from historical averages. Investors' views then tilt those returns
in proportion to their confidence. This reproduces He and Litterman's
(1999) seven-country example: the view that German equities will beat
the rest of Europe by 5% moves weight into Germany, out of France and
the UK, and leaves the other countries' weights alone.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.portfolio import black_litterman, implied_equilibrium_returns

names = ["Australia", "Canada", "France", "Germany", "Japan", "UK", "USA"]
corr = np.array(
    [
        [1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
        [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
        [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
        [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
        [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
        [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
        [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000],
    ]
)
vol = np.array([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187])
cov = corr * np.outer(vol, vol)
w_mkt = np.array([0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615])

# %%
# Reverse optimization
# --------------------

pi = implied_equilibrium_returns(cov, w_mkt, delta=2.5)
print("equilibrium returns (%):", np.round(100 * pi, 1))

# %%
# One relative view
# -----------------

P = np.array([[0.0, 0.0, -0.295, 1.0, 0.0, -0.705, 0.0]])
res = black_litterman(cov, w_mkt, P, [0.05], tau=0.05, delta=2.5)
x = np.arange(7)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
ax1.bar(x - 0.2, 100 * res.prior_returns, 0.4, label="equilibrium")
ax1.bar(x + 0.2, 100 * res.posterior_returns, 0.4, label="Black-Litterman")
ax1.set_ylabel("expected excess return (%)")
ax1.legend()
ax2.bar(x - 0.2, 100 * w_mkt, 0.4, label="market")
ax2.bar(x + 0.2, 100 * res.weights, 0.4, label="optimal with the view")
ax2.axhline(0, color="0.3", lw=0.8)
ax2.set_xticks(x, names)
ax2.set_ylabel("weight (%)")
ax2.legend()
fig.tight_layout()
print("posterior returns (%):", np.round(100 * res.posterior_returns, 1))
print("optimal weights (%):  ", np.round(100 * res.weights, 1))

# %%
# Confidence in the view
# ----------------------
#
# Scaling :math:`\Omega` moves the German weight continuously between
# the market weight (no confidence) and the weight that makes the view
# hold exactly.

scales = np.logspace(-3, 3, 40)
base = (P @ (0.05 * cov) @ P.T)[0, 0]
germany = [black_litterman(cov, w_mkt, P, [0.05], omega=[[s * base]]).weights[3] for s in scales]
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.semilogx(scales, 100 * np.array(germany))
ax.axhline(100 * w_mkt[3] / 1.05, color="0.4", ls=":", label="market weight / (1 + tau)")
ax.set_xlabel("view uncertainty relative to He-Litterman's default")
ax.set_ylabel("Germany weight (%)")
ax.legend()
fig.tight_layout()
plt.show()
