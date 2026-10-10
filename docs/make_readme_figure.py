"""Regenerate the README hero figure: python docs/make_readme_figure.py

Writes docs/source/_static/images/readme_hero.png, which README.md embeds by
its raw.githubusercontent.com URL so it also renders on PyPI.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import CRRTree, black_scholes_price
from financialkit.portfolio import efficient_frontier
from financialkit.portfolio.visualizers import plot_frontier
from financialkit.stochastic_volatility import HestonModel
from financialkit.stochastic_volatility.visualizers import plot_smile

OUT = Path(__file__).parent / "source" / "_static" / "images" / "readme_hero.png"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

# The CRR binomial tree oscillating into the Black-Scholes price.
S, K, T, r, sigma = 100.0, 95.0, 1.0, 0.05, 0.2
steps = np.arange(5, 201)
ax1.plot(steps, [CRRTree(n).price(S, K, T, r, sigma).price for n in steps], lw=1, label="CRR tree")
ax1.axhline(black_scholes_price(S, K, T, r, sigma), color="k", ls=":", label="Black-Scholes")
ax1.set_xlabel("time steps n")
ax1.set_ylabel("call price")
ax1.set_title("A binomial tree converging to Black-Scholes")
ax1.legend()

# Heston's correlation between spot and variance tilts the smile into a skew.
strikes = np.linspace(70, 135, 27)
forward = 100.0 * np.exp(0.02)
for rho in (-0.7, 0.0, 0.7):
    vols = HestonModel(0.04, 1.5, 0.04, 0.5, rho).implied_vol(100.0, strikes, 1.0, 0.02)
    plot_smile(strikes, vols, forward=forward, ax=ax2, label=f"rho = {rho}")
ax2.set_title("Heston stochastic volatility: smile and skew")
ax2.legend()

# Markowitz's frontier, Tobin's tangency portfolio and the capital market line.
names = ["bonds", "credit", "US equity", "intl equity", "emerging"]
mu = np.array([0.03, 0.045, 0.07, 0.075, 0.09])
vol = np.array([0.05, 0.08, 0.16, 0.18, 0.25])
corr = np.array(
    [
        [1.0, 0.6, -0.1, -0.1, 0.0],
        [0.6, 1.0, 0.3, 0.3, 0.4],
        [-0.1, 0.3, 1.0, 0.8, 0.7],
        [-0.1, 0.3, 0.8, 1.0, 0.75],
        [0.0, 0.4, 0.7, 0.75, 1.0],
    ]
)
plot_frontier(efficient_frontier(mu, corr * np.outer(vol, vol), 60, max_return=0.12), ax=ax3, labels=names, rf=0.02)
ax3.set_title("The efficient frontier and the tangency portfolio")

fig.savefig(OUT, dpi=110)
print(f"wrote {OUT}")
