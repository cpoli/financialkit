:orphan:

One Option, Five Pricers
========================

A European call has a closed-form price, which makes it the perfect test
case for every numerical method in quantitative finance. This tutorial
prices the same option five ways, with five of financialkit's domains:
the Black-Scholes formula from :mod:`financialkit.options`, binomial and
trinomial trees from the same package, Monte Carlo from
:mod:`financialkit.monte_carlo`, finite differences from
:mod:`financialkit.pde_methods`, and Fourier inversion from
:mod:`financialkit.stochastic_volatility`. Each method is checked against
the formula, and each fails, or converges, in its own characteristic way.

The option: a nine-month call struck at 105 on a stock trading at 100,
with a 4% rate and 25% volatility.

.. code-block:: python

   import numpy as np
   import financialkit as fk

   S, K, T, r, sigma = 100.0, 105.0, 0.75, 0.04, 0.25
   exact = fk.options.black_scholes_price(S, K, T, r, sigma)
   print(f"Black-Scholes: {exact:.6f}")
   # Black-Scholes: 7.786503

1. Trees: convergence that oscillates
-------------------------------------

The Cox-Ross-Rubinstein tree converges to Black-Scholes at rate
:math:`1/n`, but not smoothly: the error flips sign between odd and even
step counts, because the strike falls at a different place relative to the
tree's terminal nodes. The Leisen-Reimer tree centres its nodes on the
strike and removes most of the oscillation:

.. code-block:: python

   for n in (100, 101, 1000):
       err = fk.options.CRRTree(n).price(S, K, T, r, sigma).price - exact
       print(f"CRR, {n:4d} steps: error {err:+.5f}")
   # CRR,  100 steps: error -0.00387
   # CRR,  101 steps: error +0.01716
   # CRR, 1000 steps: error +0.00196
   lr = fk.options.LeisenReimerTree(101).price(S, K, T, r, sigma).price
   print(f"Leisen-Reimer, 101 steps: error {lr - exact:+.2e}")
   # Leisen-Reimer, 101 steps: error -4.22e-05

2. Monte Carlo: an estimate with an error bar
---------------------------------------------

Monte Carlo never returns *the* price, only an estimate with a standard
error that shrinks like :math:`1/\sqrt{n}`. The honest check is whether
the confidence interval covers the exact value:

.. code-block:: python

   mc = fk.monte_carlo.european_option_mc(S, K, T, r, sigma, n_paths=400_000, rng=2024)
   print(f"Monte Carlo: {mc.estimate:.4f} +/- {mc.std_error:.4f}")
   # Monte Carlo: 7.8087 +/- 0.0218
   print(f"95% interval {mc.ci[0]:.4f} to {mc.ci[1]:.4f}; covers the exact price: {mc.covers(exact)}")
   # 95% interval 7.7659 to 7.8515; covers the exact price: True
   print(f"distance from Black-Scholes: {(mc.estimate - exact) / mc.std_error:.2f} standard errors")
   # distance from Black-Scholes: 1.02 standard errors

Four times as many paths would only halve the error bar. The variance
reduction techniques in :mod:`financialkit.monte_carlo` (antithetic and
control variates, stratification, quasi-Monte Carlo) shrink it without
the extra paths.

3. Finite differences: second-order convergence
-----------------------------------------------

The Crank-Nicolson scheme solves the Black-Scholes PDE backwards from the
payoff. Two fully implicit Rannacher steps at the start damp the
oscillations that the payoff's kink would otherwise cause, and the scheme
is then second order: doubling the grid in both directions cuts the error
by about four.

.. code-block:: python

   for n in (100, 200, 400):
       pde = fk.pde_methods.CrankNicolson(n, n // 2, rannacher_steps=2).price(S, K, T, r, sigma)
       print(f"Crank-Nicolson, {n} x {n // 2} grid: error {pde.price - exact:+.5f}")
   # Crank-Nicolson, 100 x 50 grid: error -0.03810
   # Crank-Nicolson, 200 x 100 grid: error -0.00948
   # Crank-Nicolson, 400 x 200 grid: error -0.00237

4. Fourier inversion: exact to rounding
---------------------------------------

The Fourier methods need only the characteristic function of the log
price, which is why they extend to Heston, variance gamma and other models
with no closed-form price. Under Black-Scholes they reproduce the formula
almost exactly, and one FFT prices a whole strip of strikes:

.. code-block:: python

   model = fk.stochastic_volatility.BlackScholesModel(sigma)
   cm = fk.stochastic_volatility.carr_madan_price(model, S, K, T, r)
   cos = fk.stochastic_volatility.cos_price(model, S, K, T, r)
   print(f"Carr-Madan FFT: error {cm - exact:+.1e}")
   # Carr-Madan FFT: error -2.2e-07
   print(f"COS:            error {cos - exact:+.1e}")
   # COS:            error -1.8e-15

   strikes = np.linspace(80.0, 120.0, 9)
   strip = fk.stochastic_volatility.carr_madan_price(model, S, strikes, T, r)
   print(f"worst error over 9 strikes: {np.abs(strip - fk.options.black_scholes_price(S, strikes, T, r, sigma)).max():.1e}")
   # worst error over 9 strikes: 2.2e-07

The scorecard
-------------

At their finest settings above, all five agree with the formula to within
a cent, and Monte Carlo to within two standard errors:

.. list-table::
   :header-rows: 1

   * - Method
     - Error behaviour
     - Strength
   * - Black-Scholes formula
     - exact
     - instant, but European options only
   * - Binomial tree
     - :math:`O(1/n)`, oscillating
     - early exercise is trivial
   * - Monte Carlo
     - :math:`O(1/\sqrt{n})`, random
     - path dependence and many factors
   * - Crank-Nicolson
     - :math:`O(\Delta S^2 + \Delta t^2)`
     - the whole surface :math:`V(S, t)` at once
   * - Fourier (FFT, COS)
     - spectral
     - any model with a characteristic function

When the formula runs out: the American put
-------------------------------------------

An American put has no closed form, so the methods now check each other.
The tree handles early exercise by comparing values at each node, the PDE
by projected SOR at each time step, and Longstaff-Schwartz by regressing
continuation values across simulated paths. The Barone-Adesi-Whaley
formula is a fast analytic approximation:

.. code-block:: python

   tree = fk.options.CRRTree(2000).price(S, K, T, r, sigma, option_type="put", american=True).price
   psor = fk.pde_methods.CrankNicolson(400, 200, rannacher_steps=2, american_method="psor").price(
       S, K, T, r, sigma, option_type="put", american=True
   ).price
   lsm = fk.exotic_options.longstaff_schwartz(S, K, T, r, sigma, n_exercise=50, n_paths=100_000, rng=7)
   baw = fk.options.barone_adesi_whaley(S, K, T, r, sigma, option_type="put").price
   european = fk.options.black_scholes_price(S, K, T, r, sigma, option_type="put")
   print(f"CRR tree, 2000 steps: {tree:.4f}")
   # CRR tree, 2000 steps: 10.1183
   print(f"PSOR finite differences: {psor:.4f}")
   # PSOR finite differences: 10.1153
   print(f"Longstaff-Schwartz: {lsm.estimate:.4f} +/- {lsm.std_error:.4f}")
   # Longstaff-Schwartz: 10.1092 +/- 0.0134
   print(f"Barone-Adesi-Whaley: {baw:.4f}")
   # Barone-Adesi-Whaley: 10.0877
   print(f"early-exercise premium: {tree - european:.4f}")
   # early-exercise premium: 0.4350

The tree and PSOR agree to within a cent. Longstaff-Schwartz exercises on
a 50-date grid with an estimated rule, so it is biased slightly low, yet
it still lands within one standard error of the tree. Barone-Adesi-Whaley
is an approximation, and it is three cents low.

See Also
--------

- :doc:`/api/options`
- :doc:`/api/monte_carlo`
- :doc:`/api/pde_methods`
- :doc:`/api/stochastic_volatility`
- :doc:`/history/options_breakthroughs`
- :doc:`/history/pde_methods_breakthroughs`
