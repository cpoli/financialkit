r"""
Ikeda-Kunitomo double barriers
==============================

A double knock-out dies if the price leaves a corridor :math:`(L, U)`.
Naoto Ikeda and Masaaki Kunitomo (1992) solved it, even for curved
(exponential) barriers, with an infinite series of images reflected
alternately through both boundaries. The terms decay like
:math:`e^{-2n^2\ln^2(U/L)/\sigma^2T}`, so two or three are plenty unless
the corridor is very narrow. Unlike a vanilla call, a double knock-out
call loses value as volatility rises: more paths hit a barrier.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.exotic_options import double_barrier_price, plot_barrier_paths
from financialkit.integrators import gbm_paths

K, L, U, T, r = 100.0, 80.0, 130.0, 0.5, 0.05

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
t, paths = gbm_paths(100.0, r, 0.25, T, 100, 40, rng=0)
plot_barrier_paths(t, paths, [L, U], ax=axes[0])

# %%
# The corridor value and its volatility
# -------------------------------------

S = np.linspace(L, U, 300)
for sigma in (0.1, 0.2, 0.3, 0.4):
    axes[1].plot(S, double_barrier_price(S, K, L, U, T, r, sigma), label=rf"$\sigma$ = {sigma:.0%}")
# The value vanishes on both barriers.
assert np.allclose(double_barrier_price(np.array([L, U]), K, L, U, T, r, 0.3), 0.0, atol=1e-12)
axes[1].set_xlabel("spot S")
axes[1].set_ylabel("double knock-out call")
axes[1].legend()

# %%
# How many images?
# ----------------
# The error of the series truncated at :math:`|n| \le N`, against :math:`N = 20`.

for width, (lo, hi) in (("wide", (60.0, 160.0)), ("medium", (80.0, 130.0)), ("narrow", (95.0, 110.0))):
    ref = double_barrier_price(100.0, 100.0, lo, hi, T, r, 0.3, n_terms=20)
    err = [abs(double_barrier_price(100.0, 100.0, lo, hi, T, r, 0.3, n_terms=n) - ref) + 1e-17 for n in range(0, 6)]
    axes[2].semilogy(range(0, 6), err, "o-", label=f"{width}: ({lo:g}, {hi:g})")
    # Five images already match twenty to 1e-10, even in the narrow corridor.
    assert err[5] < 1e-10
axes[2].set_xlabel("terms N")
axes[2].set_ylabel("truncation error")
axes[2].legend()
fig.tight_layout()
plt.show()
