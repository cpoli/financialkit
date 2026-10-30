r"""
Craig-Sneyd ADI with mixed derivatives
======================================

Peaceman and Rachford's scheme has no room for a mixed derivative, and
two correlated assets bring one, :math:`\rho\sigma_1\sigma_2 V_{xy}`.
Taking it explicitly keeps each half step tridiagonal but costs the
scheme its second order. Ian Craig and Alfred Sneyd (1988) kept the
one-direction implicit sweeps of the Douglas scheme and followed them
with a corrector that re-evaluates the mixed term at the new time level
and sweeps again, which restores second order in time for any
correlation. :class:`~financialkit.pde_methods.CraigSneydADI` prices
Margrabe's exchange option, and the time-step study compares it with the
Douglas scheme it corrects and with Peaceman-Rachford's explicit
mixed term.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import black_scholes_price
from financialkit.pde_methods import CraigSneydADI, PeacemanRachfordADI

S1, S2, T, r, sigma1, sigma2, rho = 100.0, 95.0, 1.0, 0.05, 0.2, 0.3, 0.5


def exchange(s1, s2):
    return np.maximum(s1 - s2, 0.0)


margrabe = black_scholes_price(S1, S2, T, 0.0, np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2))
res = CraigSneydADI(120, 120, 40).price(S1, S2, T, r, sigma1, sigma2, rho, exchange)
print(f"Craig-Sneyd {res.price:.4f}, Margrabe {margrabe:.4f}")
assert abs(res.price - margrabe) < 0.01

# %%
# Order in time
# -------------
#
# On a fixed 80 x 80 grid the error against a 640-step solve isolates
# the time stepping. Douglas and the explicit mixed term of
# Peaceman-Rachford halve the error when the step halves; Craig-Sneyd
# quarters it.

steps = np.array([5, 10, 20, 40, 80])
ref = CraigSneydADI(80, 80, 640).price(S1, S2, T, r, sigma1, sigma2, rho, exchange).price
errors = {
    "Douglas": [CraigSneydADI(80, 80, n, scheme="douglas").price(S1, S2, T, r, sigma1, sigma2, rho, exchange).price for n in steps],
    "Peaceman-Rachford, explicit V_xy": [PeacemanRachfordADI(80, 80, n).price(S1, S2, T, r, sigma1, sigma2, rho, exchange).price for n in steps],
    "Craig-Sneyd": [CraigSneydADI(80, 80, n).price(S1, S2, T, r, sigma1, sigma2, rho, exchange).price for n in steps],
}
errors = {name: np.abs(np.array(p) - ref) for name, p in errors.items()}
for name, e in errors.items():
    order = np.log2(e[-2] / e[-1])
    print(f"{name:34s} observed order {order:.2f}")
assert np.log2(errors["Douglas"][-2] / errors["Douglas"][-1]) < 1.3
assert np.log2(errors["Craig-Sneyd"][-2] / errors["Craig-Sneyd"][-1]) > 1.7

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for name, e in errors.items():
    axes[0].loglog(steps, e, "o-", label=name)
axes[0].loglog(steps, 0.5 / steps**2, "k:", lw=1, label="slope -2")
axes[0].set_xlabel("time steps")
axes[0].set_ylabel("|error| against 640 steps")
axes[0].legend(fontsize=8)
axes[0].set_title("rho = 0.5")

# %%
# Across correlations
# -------------------
#
# With 20 steps the corrector keeps Craig-Sneyd close to Margrabe's price
# as the correlation, and with it the mixed term, grows.

rhos = np.linspace(-0.8, 0.8, 9)
exact = black_scholes_price(S1, S2, T, 0.0, np.sqrt(sigma1**2 + sigma2**2 - 2 * rhos * sigma1 * sigma2))
cs = [CraigSneydADI(100, 100, 20).price(S1, S2, T, r, sigma1, sigma2, c, exchange).price for c in rhos]
do = [CraigSneydADI(100, 100, 20, scheme="douglas").price(S1, S2, T, r, sigma1, sigma2, c, exchange).price for c in rhos]
assert np.max(np.abs(np.array(cs) - exact)) < np.max(np.abs(np.array(do) - exact))
axes[1].plot(rhos, np.array(do) - exact, "s--", label="Douglas")
axes[1].plot(rhos, np.array(cs) - exact, "o-", label="Craig-Sneyd")
axes[1].axhline(0.0, color="k", lw=0.5)
axes[1].set_xlabel("correlation rho")
axes[1].set_ylabel("price - Margrabe")
axes[1].legend()
axes[1].set_title("20 time steps, 100 x 100 grid")
fig.tight_layout()
plt.show()
