:orphan:

The Yield Curve, End to End
===========================

Every interest-rate calculation starts from a discount curve, but markets
quote par rates, not discount factors. This tutorial follows one curve
from quotes to derivatives across two domains: bootstrapping and
smoothing it in :mod:`financialkit.fixed_income`, pricing a swap on it,
fitting a Hull-White short-rate model to it in
:mod:`financialkit.interest_rate_models`, checking that model by Monte
Carlo through :mod:`financialkit.integrators`, and finally pricing a cap
two ways.

1. Bootstrapping par rates into zero rates
------------------------------------------

Par rates are the coupons at which bonds of each maturity would trade at
par. Bootstrapping solves for discount factors one maturity at a time,
each using the ones already found:

.. code-block:: python

   import numpy as np
   import financialkit as fk

   maturities = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30.0])
   par_rates = np.array([3.10, 3.20, 3.35, 3.50, 3.70, 3.85, 4.00, 4.25, 4.30]) / 100
   curve = fk.fixed_income.bootstrap_par_curve(maturities, par_rates)

   t = np.array([1.0, 5.0, 10.0, 30.0])
   print("zero rates (%):", np.round(100 * curve.zero_rate(t), 3))
   # zero rates (%): [3.175 3.683 4.006 4.356]
   print("discount factors:", np.round(curve.discount(t), 5))
   # discount factors: [0.96874 0.83179 0.66994 0.2707 ]
   print(f"repriced 10-year par yield: {100 * curve.par_yield(10.0):.4f}%")
   # repriced 10-year par yield: 4.0000%

The curve reprices its own inputs exactly. Zero rates follow the
package-wide convention of continuous compounding, so the 1-year zero
rate of 3.175% is just the 3.20% semiannual par rate restated. Further
out the two part ways: on an upward-sloping curve the 10-year zero rate
(4.006%) sits above the continuously compounded 10-year par rate
(3.961%), because a par bond's early coupons are discounted at the lower
short rates.

2. Smoothing with Nelson-Siegel
-------------------------------

A bootstrapped curve passes through every quote, noise included. Nelson
and Siegel's four-parameter curve (level, slope, curvature and a decay
time) is what central banks publish instead:

.. code-block:: python

   fit = fk.fixed_income.fit_nelson_siegel(maturities, curve.zero_rate(maturities))
   b0, b1, b2, tau = fit.params
   print(f"level {100 * b0:.2f}%, slope {100 * b1:.2f}%, curvature {100 * b2:.2f}%, tau {tau:.2f} years")
   # level 4.43%, slope -1.44%, curvature 1.20%, tau 6.93 years
   print(f"RMSE against the bootstrapped curve: {1e4 * fit.rmse:.1f} bp")
   # RMSE against the bootstrapped curve: 1.1 bp

Four numbers summarise the whole term structure to about a basis point.

3. A swap on the curve
----------------------

The par swap rate is the fixed rate that gives a swap zero value. Paying
fixed below it is worth money:

.. code-block:: python

   print(f"5-year par swap rate: {100 * fk.fixed_income.swap_rate(curve, 5.0):.4f}%")
   # 5-year par swap rate: 3.7000%
   swap = fk.fixed_income.interest_rate_swap_value(curve, 10_000_000, 0.035, 5.0)
   print(f"pay 3.50% fixed on 10 million for 5 years: worth {swap.value:,.0f}")
   # pay 3.50% fixed on 10 million for 5 years: worth 90,923

With semiannual payments the 5-year swap rate equals the 5-year par yield,
3.70%: a swap is a par bond against a floating-rate note.

4. A short-rate model fitted to the curve
-----------------------------------------

Pricing options on rates needs a model of how the curve moves. Hull and
White made Vasicek's mean-reverting short rate fit today's curve exactly,
by letting its long-run level drift with time:

.. code-block:: python

   hw = fk.interest_rate_models.HullWhiteModel(curve, a=0.1, sigma=0.01)
   print("model bond prices:", np.round(hw.zero_coupon_bond(t), 5))
   # model bond prices: [0.96874 0.83179 0.66994 0.2707 ]
   print("max mismatch with the curve:", f"{np.abs(hw.zero_coupon_bond(t) - curve.discount(t)).max():.1e}")
   # max mismatch with the curve: 0.0e+00

The closed-form bond prices match the curve by construction. A stronger
check simulates the short rate and averages discount factors along each
path (the Feynman-Kac representation). The paths come from the shared
SDE schemes in :mod:`financialkit.integrators`:

.. code-block:: python

   mc = fk.interest_rate_models.monte_carlo_bond_prices(hw, [1.0, 2.0, 5.0, 10.0], n_steps=200, n_paths=20_000, rng=2)
   print("Monte Carlo:", np.round(mc.prices, 5))
   # Monte Carlo: [0.96873 0.93564 0.83159 0.66898]
   print("(MC - exact) / SE:", np.round(mc.z_scores, 2))
   # (MC - exact) / SE: [-0.37 -0.11 -0.64 -1.57]

Every z-score is within two standard errors, so the simulation agrees
with the closed form.

5. A cap, two ways
------------------

A 5-year cap at 4% pays whenever 3-month rates fix above 4%. The market
convention prices each caplet with Black-76, the single implementation
in :mod:`financialkit.options` that the rates domain calls, using a
lognormal forward with a quoted volatility. Hull-White prices the same
cap from its own normal dynamics:

.. code-block:: python

   black = fk.interest_rate_models.cap_floor_price(curve, 0.04, 5.0, vol=0.20)
   model = fk.interest_rate_models.model_cap_floor_price(hw, 0.04, 5.0)
   print(f"Black-76, 20% vol: {1e4 * black.price:.1f} bp of notional")
   # Black-76, 20% vol: 159.8 bp of notional
   print(f"Hull-White, a = 0.1, sigma = 1%: {1e4 * model.price:.1f} bp of notional")
   # Hull-White, a = 0.1, sigma = 1%: 179.8 bp of notional

The two disagree because a 20% lognormal vol and a 1% normal vol are
different assumptions. Calibrating a model means choosing ``a`` and
``sigma`` so that its cap prices match the market's Black quotes, after
which it can price instruments with no market quote, such as Bermudan
swaptions.

See Also
--------

- :doc:`/api/fixed_income`
- :doc:`/api/interest_rate_models`
- :doc:`/history/fixed_income_breakthroughs`
- :doc:`/history/interest_rate_models_breakthroughs`
