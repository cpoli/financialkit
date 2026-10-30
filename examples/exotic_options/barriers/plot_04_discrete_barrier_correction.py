r"""
Broadie-Glasserman-Kou discrete barriers
========================================

Real barrier contracts check the price once a day, not continuously, and
a path can slip through the barrier between two checks. Mark Broadie,
Paul Glasserman and Steven Kou (1997) showed that the discretely
monitored option is, to :math:`o(1/\sqrt m)`, the continuous one with
the barrier shifted away from the spot:

.. math:: H \to H e^{\pm \beta\sigma\sqrt{T/m}}, \qquad \beta = -\zeta(1/2)/\sqrt{2\pi} \approx 0.5826.

Monte Carlo with exactly :math:`m` monitoring dates checks the correction.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.exotic_options import BGK_BETA, barrier_option_mc, barrier_price, discrete_barrier_price

S, K, H, T, r, sigma = 100.0, 100.0, 85.0, 1.0, 0.05, 0.3
print(f"beta = {BGK_BETA:.6f}")
monitors = np.array([4, 12, 26, 52, 252])
mc = [barrier_option_mc(S, K, H, T, r, sigma, n_steps=int(m), n_paths=40_000, continuous=False, rng=int(m)) for m in monitors]
continuous = barrier_price(S, K, H, T, r, sigma)
corrected = np.array([discrete_barrier_price(S, K, H, T, r, sigma, n_monitor=int(m)) for m in monitors])

# %%
# Discrete monitoring, the continuous formula and the shifted barrier
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(monitors, [m.estimate for m in mc], yerr=[2 * m.std_error for m in mc], fmt="o", capsize=3, label="Monte Carlo, m dates")
ax.semilogx(monitors, corrected, "s--", label="continuous formula, shifted barrier")
ax.axhline(continuous, color="k", lw=1, label="continuous monitoring")
ax.set_xlabel("monitoring dates m")
ax.set_ylabel("down-and-out call")
ax.legend()
for m, res, c in zip(monitors, mc, corrected, strict=True):
    print(f"m = {m:4d}: MC {res.estimate:.4f} +/- {res.std_error:.4f}, BGK {c:.4f}, continuous {continuous:.4f}")
    # The shifted barrier matches discrete monitoring within four standard
    # errors, and discrete monitoring is worth more than continuous.
    assert abs(res.estimate - c) < 4 * res.std_error
    assert c > continuous
plt.show()
