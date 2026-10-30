r"""
Hull-White trinomial tree: callable bonds and the option-adjusted spread
=======================================================================

Hull and White (1994) built their model as a trinomial tree. The rate
deviation sits on a grid :math:`j\Delta x`, :math:`\Delta x = \sigma\sqrt{3\Delta t}`,
its branch probabilities match the mean reversion, and at
:math:`|j| = j_{max}` the branching turns inward, so the tree stops
widening. A shift per time step, fitted by forward induction, makes the
tree reprice today's curve.

A callable bond is valued by rolling back on that tree, taking on every
call date the smaller of the bond's value and the call price. The
option-adjusted spread (OAS) is the spread over the tree's rates at which
the model matches the market price: the bond's spread with the
issuer's call priced out.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import NelsonSiegelCurve, ShiftedCurve
from financialkit.fixed_income.core.base import coupon_times
from financialkit.interest_rate_models import HullWhiteModel, callable_bond_price, coupon_bond_option, hull_white_tree, option_adjusted_spread

curve = NelsonSiegelCurve(0.05, -0.02, 0.01, 2.0)
a, sigma = 0.1, 0.01

# %%
# The tree reprices the curve, and converges to Jamshidian's formula
# ------------------------------------------------------------------
# A 10-year 5% semiannual bond callable once, at par in 5 years, is the
# straight bond minus a European call on the bond, which Jamshidian's
# decomposition prices in closed form.

times = coupon_times(10.0, 2)
flows = np.full(times.size, 0.025)
flows[-1] += 1.0
after = times > 5.0
european = coupon_bond_option(HullWhiteModel(curve, a, sigma), 5.0, times[after], flows[after], 1.0, "call").price
exact = float(np.sum(flows * curve.discount(times))) - european

steps = np.array([20, 40, 80, 160, 320, 640])
errors = []
for n in steps:
    tree = hull_white_tree(curve, a, sigma, 10.0, n)
    errors.append(callable_bond_price(tree, 10.0, 0.05, first_call=5.0, last_call=5.0).price - exact)
    assert abs(tree.zero_coupon_bond(10.0) - curve.discount(10.0)) < 1e-12
errors = np.array(errors)
print(f"j_max = {tree.j_max} with {steps[-1]} steps; error at 640 steps = {errors[-1]:.1e}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].loglog(steps, np.abs(errors), "o-")
axes[0].loglog(steps, np.abs(errors[0]) * steps[0] / steps, ":", color="gray", label="first order")
axes[0].set_xlabel("time steps")
axes[0].set_ylabel("|tree - Jamshidian|")
axes[0].set_title("European call: the tree converges")
axes[0].legend()

# %%
# Negative convexity: the call caps the price
# -------------------------------------------
# Callable every half year after a 2-year lockout. When rates fall the
# straight bond keeps rising, but the issuer would call, so the callable
# bond flattens out near the call price.

shifts = np.linspace(-0.03, 0.03, 13)
straight, callable_ = [], []
for s in shifts:
    tree = hull_white_tree(ShiftedCurve(curve, s), a, sigma, 10.0, 80)
    res = callable_bond_price(tree, 10.0, 0.05, first_call=2.0, face=100.0)
    straight.append(res.straight_price)
    callable_.append(res.price)
straight, callable_ = np.array(straight), np.array(callable_)
axes[1].plot(100 * shifts, straight, label="straight bond")
axes[1].plot(100 * shifts, callable_, label="callable at par after 2 years")
axes[1].axhline(100, color="gray", lw=0.8, ls=":")
axes[1].set_xlabel("parallel curve shift (%)")
axes[1].set_ylabel("price")
axes[1].set_title("Price compression")
axes[1].legend()

# %%
# OAS against Z-spread
# --------------------
# Suppose the callable bond trades at 97. Its Z-spread (the spread on its
# straight cash flows) looks generous, but part of it is pay for the
# call the investor has sold. The OAS removes that part, and it depends on
# the volatility at which the call is valued: more volatility, a dearer
# call, a smaller OAS.

market = 97.0
vols = np.linspace(0.004, 0.02, 9)
tree = hull_white_tree(curve, a, sigma, 10.0, 80)
z = option_adjusted_spread(tree, market, 10.0, 0.05, call_price=1e6, face=100.0)  # a call never exercised
oas = np.array([option_adjusted_spread(hull_white_tree(curve, a, v, 10.0, 80), market, 10.0, 0.05, first_call=2.0, face=100.0) for v in vols])
print(f"Z-spread = {1e4 * z:.0f} bp; OAS at sigma = 1% = {1e4 * np.interp(0.01, vols, oas):.0f} bp")
axes[2].plot(100 * vols, 1e4 * oas, "o-", label="OAS")
axes[2].axhline(1e4 * z, color="C3", ls="--", label="Z-spread")
axes[2].set_xlabel("short-rate volatility sigma (%)")
axes[2].set_ylabel("spread (bp)")
axes[2].set_title("A callable bond priced at 97")
axes[2].legend()
fig.tight_layout()

assert np.abs(errors[-1]) < 1e-4 and np.abs(errors[-1]) < np.abs(errors[0]) / 10  # converges to the closed form
assert np.all(callable_ < straight) and np.all(np.diff(callable_) < 0)
assert callable_[0] - callable_[6] < 0.4 * (straight[0] - straight[6])  # the call caps the rally
assert np.all(np.diff(oas) < 0) and np.all(oas < z)  # OAS below the Z-spread, falling with volatility

plt.show()
