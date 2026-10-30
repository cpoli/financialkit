r"""
Israel, Rosenthal and Wei: finding a valid generator
====================================================

Going from an annual transition matrix :math:`P` to a generator is not
as simple as :math:`\log P`. Israel, Rosenthal and Wei (2001) showed that
for empirical rating matrices the principal logarithm usually has
negative off-diagonal entries, which would be negative migration
intensities. Their fix: set those entries to zero and absorb the change
in the diagonal. A weighted variant spreads it over the row instead.
Both leave :math:`e^{G}` within a few basis points of :math:`P`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.credit import RATINGS, creditmetrics_matrix, generator_from_transition, is_valid_generator, regularize_generator, transition_from_generator
from financialkit.credit.visualizers import plot_transition_matrix

P = creditmetrics_matrix()
L = generator_from_transition(P)
off = L - np.diag(np.diag(L))
print(f"log(P) is a valid generator: {is_valid_generator(L)}; {np.sum(off < 0)} negative off-diagonal entries, most negative {off.min():.2e}")
assert not is_valid_generator(L)
ax = plot_transition_matrix(L, labels=RATINGS, fmt="{:.2f}")
ax.set_title("log(P) in % per year: negative intensities in red")

# %%
# The two adjustments
# -------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, method in zip(axes, ("diagonal", "weighted"), strict=True):
    G = regularize_generator(L, method)
    err = transition_from_generator(G) - P
    plot_transition_matrix(err, labels=RATINGS, ax=ax, fmt="{:.2f}", scale=1e4)
    ax.set_title(f"{method}: exp(G) - P in bp (max {1e4 * np.abs(err).max():.1f})")
    print(f"{method:8s}: valid {is_valid_generator(G)}, max |exp(G) - P| = {np.abs(err).max():.2e}")
    assert is_valid_generator(G) and np.abs(err).max() < 5e-4
fig.tight_layout()

plt.show()
