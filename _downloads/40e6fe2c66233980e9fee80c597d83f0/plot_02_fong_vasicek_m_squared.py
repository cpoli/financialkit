r"""
Fong-Vasicek M-squared
======================

Redington's conditions protect against parallel shifts only. Fong and
Vasicek (1984) showed that when the forward curve twists with slope at
most :math:`K`, a duration-matched portfolio loses at most

.. math:: \frac{\Delta V_H}{V_H} \ge -\tfrac12 K M^2, \qquad M^2 = \sum_i w_i (t_i - H)^2,

where :math:`M^2` measures how far the cash flows are spread around the
horizon :math:`H`. Among immunized portfolios the bullet, with the
smallest :math:`M^2`, is the least exposed to twists.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import (
    CashFlows,
    FlatCurve,
    bond_cash_flows,
    fong_vasicek_bound,
    horizon_value,
    linear_forward_shock,
    m_squared,
    redington_immunize,
)

curve, H = FlatCurve(0.04), 8.0
target = CashFlows([H], [1000.0])
portfolios = {
    "bullet 7y/9y": redington_immunize(target, [bond_cash_flows(1.0, 0.0, 7.0), bond_cash_flows(1.0, 0.0, 9.0)], curve).portfolio,
    "ladder 4y/14y": redington_immunize(target, [bond_cash_flows(1.0, 0.0, 4.0), bond_cash_flows(1.0, 0.0, 14.0)], curve).portfolio,
    "barbell 1y/30y": redington_immunize(target, [bond_cash_flows(1.0, 0.0, 1.0), bond_cash_flows(1.0, 0.0, 30.0)], curve).portfolio,
}

# %%
# Horizon value under twists of the forward curve
# -----------------------------------------------

slopes = np.linspace(-0.004, 0.004, 41)
fig, ax = plt.subplots()
for name, port in portfolios.items():
    m2 = m_squared(port, H, curve)
    base = horizon_value(port, curve, H)
    change = np.array([horizon_value(port, curve.shifted(linear_forward_shock(0.0, b, H)), H) / base - 1 for b in slopes])
    (line,) = ax.plot(1e4 * slopes, 100 * change, label=f"{name}, $M^2$ = {m2:.1f}")
    ax.plot(1e4 * slopes[slopes > 0], 100 * fong_vasicek_bound(m2, slopes[slopes > 0]), ls=":", color=line.get_color())
ax.axhline(0.0, color="0.6", lw=0.8)
ax.set_xlabel("forward-curve slope change $b$ (bp per year)")
ax.set_ylabel("change in horizon value (%)")
ax.set_title("Dotted: Fong-Vasicek bound $-K M^2/2$")
ax.legend()
plt.show()
