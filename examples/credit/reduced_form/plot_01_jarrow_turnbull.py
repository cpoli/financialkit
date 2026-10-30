r"""
Jarrow and Turnbull: hazard rates and survival curves
=====================================================

Jarrow and Turnbull (1995) modelled the timing of default rather than
its cause: default is the first jump of a Poisson process with
intensity :math:`\lambda(t)`, so the survival probability is
:math:`Q(t) = e^{-\int_0^t \lambda}`. A risky zero-coupon bond that
recovers a fraction :math:`\delta` of a riskless one is worth
:math:`P(0,T)[\delta + (1-\delta)Q(T)]`, and its spread is close to
:math:`\lambda(1 - \delta)`: the credit triangle.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.credit import HazardCurve, jarrow_turnbull_zero, spread_from_hazard
from financialkit.credit.visualizers import plot_survival

curve = HazardCurve([1.0, 3.0, 5.0, 10.0], [0.010, 0.015, 0.025, 0.030])
fig, ax = plt.subplots(figsize=(7, 4))
plot_survival(curve, T=10.0, ax=ax, label="piecewise-constant hazard")
plot_survival(HazardCurve.flat(0.02), T=10.0, ax=ax, label="flat 2%", hazard=False)
ax.legend(loc="lower left")

# %%
# Simulated default times
# -----------------------
#
# Inverting the cumulative hazard of a uniform draw gives default times
# whose histogram is the default density :math:`\lambda(t) Q(t)`.

tau = curve.sample_default_times(100_000, rng=0)
q10 = float(curve.default_probability(10.0))
assert abs((tau < 10).mean() - q10) < 4 * np.sqrt(q10 * (1 - q10) / tau.size)
t = np.linspace(0, 10, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(tau[tau < 10], bins=80, density=True, weights=np.full((tau < 10).sum(), (tau < 10).mean()), color="0.7", label="simulated")
ax.plot(t, curve.default_density(t), "C3", lw=2, label="lambda(t) Q(t)")
ax.set_xlabel("default time (years)")
ax.legend()

# %%
# Risky bond prices and the credit triangle
# -----------------------------------------

T = np.linspace(0.25, 10, 40)
r, delta = 0.04, 0.4
P = jarrow_turnbull_zero(T, r, curve, delta)
assert np.allclose(P, np.exp(-r * T) * (delta + (1 - delta) * np.asarray(curve.survival(T))))
spread = -np.log(P) / T - r
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(T, 1e4 * spread, label="zero-coupon spread")
avg_hazard = np.asarray(curve.cumulative_hazard(T)) / T
triangle = np.asarray(spread_from_hazard(avg_hazard, delta))
assert np.all(np.abs(spread / triangle - 1) < 0.05)
ax.plot(T, 1e4 * triangle, "--", label="credit triangle, average hazard x (1 - R)")
ax.set_xlabel("maturity")
ax.set_ylabel("bp")
ax.legend()

plt.show()
