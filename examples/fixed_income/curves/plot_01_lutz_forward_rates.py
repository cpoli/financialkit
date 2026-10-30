r"""
Lutz: forward rates
===================

Friedrich Lutz (1940) read the term structure as a sequence of expected
future short rates. The forward rate for the period from :math:`t_1` to
:math:`t_2` is what the zero curve locks in today,

.. math:: f(t_1, t_2) = \frac{r_2 t_2 - r_1 t_1}{t_2 - t_1},

and each zero rate is the time average of the forwards before it. A rising
zero curve therefore has forwards above it, and a falling one has forwards
below it.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import ZeroCurve, forward_rates

# %%
# Hull's Table 4.5
# ----------------

tenors = np.array([1, 2, 3, 4, 5.0])
zeros = np.array([0.03, 0.04, 0.046, 0.05, 0.053])
fwd = forward_rates(tenors, zeros)
for T, r, f in zip(tenors, zeros, fwd, strict=True):
    print(f"year {T:.0f}: zero {100 * r:.2f}%  forward for year {T:.0f} {100 * f:.2f}%")
assert np.allclose(fwd, [0.03, 0.05, 0.058, 0.062, 0.065])

# %%
# Forwards above a rising curve, below a falling one
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
t = np.arange(1, 11, dtype=float)
shapes = {"rising": 0.02 + 0.03 * (1 - np.exp(-t / 3)), "inverted": 0.05 - 0.02 * (1 - np.exp(-t / 2))}
for ax, (name, r) in zip(axes, shapes.items(), strict=True):
    f = forward_rates(t, r)
    assert np.all(f >= r - 1e-12) if name == "rising" else np.all(f <= r + 1e-12)
    ax.step(np.r_[0, t], 100 * np.r_[f, f[-1]], where="post", label="one-year forwards")
    ax.plot(t, 100 * r, "o-", label="zero rates")
    ax.set_title(name)
    ax.set_xlabel("maturity (years)")
axes[0].set_ylabel("rate (%)")
axes[0].legend()

# %%
# Zero rates are time averages of forwards
# ----------------------------------------

curve = ZeroCurve(t, shapes["rising"])
f = forward_rates(t, shapes["rising"])
average_error = np.abs(np.cumsum(f) / t - curve.zero_rate(t)).max()
print("max |average of forwards - zero rate| =", average_error)
assert average_error < 1e-12
plt.show()
