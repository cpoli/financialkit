r"""
Richardson extrapolation
========================

Lewis Fry Richardson (1911), computing stresses in masonry dams by
finite differences, noticed that when the error behaves like
:math:`C h^p`, two solutions on grids :math:`h` and :math:`h/2` determine
both the constant and the limit:
:math:`V \approx V_{h/2} + (V_{h/2} - V_h)/(2^p - 1)`.
:func:`~financialkit.pde_methods.richardson_extrapolation` applies this to
Crank-Nicolson prices; three grids also confirm that the observed order
is 2, which needs the strike on a node and Rannacher's start so that the
error is a clean power of :math:`h`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import black_scholes_price
from financialkit.pde_methods import CrankNicolson, richardson_extrapolation

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
exact = black_scholes_price(S, K, T, r, sigma)

ms = np.array([20, 40, 80, 160, 320])
prices = np.array([CrankNicolson(m, m, rannacher_steps=2, s_max=400.0).price(S, K, T, r, sigma).price for m in ms])
extrapolated = [richardson_extrapolation(prices[: k + 1]).extrapolated for k in range(1, len(ms))]
res = richardson_extrapolation(prices)
print(f"observed order {res.observed_order:.3f}")
for m, p in zip(ms, prices, strict=True):
    print(f"M = N = {m:3d}: {p:.6f} (error {p - exact:+.2e})")
print(f"extrapolated:  {res.extrapolated:.6f} (error {res.extrapolated - exact:+.2e})")
assert abs(res.observed_order - 2.0) < 0.1
assert abs(res.extrapolated - exact) < abs(prices[-1] - exact) / 10

# %%
# The extrapolated sequence converges much faster
# -----------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(ms, np.abs(prices - exact), "o-", label="Crank-Nicolson")
ax.loglog(ms[1:], np.abs(np.array(extrapolated) - exact), "s-", label="Richardson from the last two")
ax.loglog(ms, np.abs(prices[0] - exact) * (ms / ms[0]) ** -2.0, ":", color="0.4", label="$M^{-2}$")
ax.set_xlabel("grid size M = N")
ax.set_ylabel("absolute error")
ax.legend()
plt.show()
