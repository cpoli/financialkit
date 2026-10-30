r"""
Jarrow, Lando and Turnbull: generator matrices
==============================================

Jarrow, Lando and Turnbull (1997) put rating migration in continuous
time. A generator :math:`\Lambda`, with non-negative off-diagonal
intensities and rows summing to zero, gives the transition matrix for
every horizon, :math:`P(t) = e^{\Lambda t}`. Default probabilities by
rating, and risky bond prices with recovery of treasury, follow for
any maturity, including fractions of a year that an annual matrix
cannot provide.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.credit import RATINGS, creditmetrics_matrix, generator_from_transition, jlt_risky_zero, regularize_generator, transition_from_generator
from financialkit.credit.visualizers import plot_transition_matrix

G = regularize_generator(generator_from_transition(creditmetrics_matrix()))
plot_transition_matrix(G, labels=RATINGS, fmt="{:.1f}")
plt.title("generator (% per year)")

# %%
# The term structure of default probabilities
# -------------------------------------------

t = np.linspace(0.0, 10.0, 81)
Pt = transition_from_generator(G, t)
fig, ax = plt.subplots(figsize=(7, 4))
for i in range(7):
    ax.semilogy(t[1:], Pt[1:, i, -1], label=RATINGS[i])
ax.set_xlabel("years")
ax.set_ylabel("cumulative default probability")
ax.legend(ncol=2)

# %%
# exp(G t) interpolates the annual matrix powers
# ----------------------------------------------

P1 = transition_from_generator(G, 1.0)
years = np.arange(1, 11)
powers = [np.linalg.matrix_power(P1, n)[3, -1] for n in years]
assert np.allclose(powers, Pt[8 * years, 3, -1], atol=1e-12)
ax.plot(years, powers, "kx", label="P(1)^n, BBB")
ax.legend(ncol=2)

# %%
# Risky zero-coupon spreads by rating
# -----------------------------------

T = np.linspace(0.25, 10, 40)
fig, ax = plt.subplots(figsize=(7, 4))
spreads = []
for i in range(1, 7):
    P = jlt_risky_zero(T, 0.03, G, i, recovery=0.4)
    spreads.append(1e4 * (-np.log(P) / T - 0.03))
    ax.plot(T, spreads[-1], label=RATINGS[i])
assert np.all(np.diff(spreads, axis=0) > 0)
ax.set_xlabel("maturity")
ax.set_ylabel("spread (bp)")
ax.set_yscale("log")
ax.legend()

plt.show()
