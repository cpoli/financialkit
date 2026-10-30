:orphan:

Where the Smile Comes From
==========================

If Black and Scholes were right, every option on a stock would imply the
same volatility. Markets disagree: implied volatility is higher for
low strikes than for high ones, a *skew*, and higher at both ends than in
the middle, a *smile*. This tutorial builds a skew from a model in
:mod:`financialkit.stochastic_volatility`, reads it back through the
implied-volatility tools in :mod:`financialkit.options`, recovers the
risk-neutral distribution it encodes, and fits the parametrization traders
actually quote.

1. Implied volatility inverts the formula
-----------------------------------------

Implied volatility is the volatility that makes Black-Scholes reproduce a
price. Run a Black-Scholes price back through it and the input comes
back, at every strike:

.. plot::
   :context: reset

   import numpy as np
   import financialkit as fk

   S, T, r = 100.0, 0.5, 0.03
   K = np.linspace(70.0, 130.0, 13)
   flat = fk.options.black_scholes_price(S, K, T, r, 0.2)
   print(np.round(fk.options.implied_volatility(flat, S, K, T, r), 6))
   # [0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2]

2. Stochastic volatility skews the smile
----------------------------------------

In Heston's model the variance follows its own mean-reverting process,
correlated with the stock. A negative correlation, which is what equity
markets show, means volatility rises when the stock falls. Price options
under Heston by Fourier inversion, then ask what volatility Black-Scholes
would need:

.. plot::
   :context: close-figs

   heston = fk.stochastic_volatility.HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7)
   prices = fk.stochastic_volatility.heston_price(heston, S, K, T, r)
   smile = fk.options.implied_vol_smile(prices, K, S, T, r)
   for k, v in zip(K[::3], smile.vols[::3]):
       print(f"K = {k:5.1f}: implied vol {100 * v:.2f}%")
   # K =  70.0: implied vol 29.39%
   # K =  85.0: implied vol 24.06%
   # K = 100.0: implied vol 18.39%
   # K = 115.0: implied vol 14.22%
   # K = 130.0: implied vol 14.28%

The long-run volatility is 20% everywhere, yet a 70 strike implies
almost 30% and a 115 strike barely 14%; at 130 the curve turns up again,
so the skew carries a smile. Before trusting a Fourier price,
check it independently: simulate Heston paths with Andersen's QE scheme
and price the at-the-money call by Monte Carlo:

.. plot::
   :context: close-figs

   mc = fk.stochastic_volatility.heston_mc_price(heston, S, 100.0, T, r, n_steps=100, n_paths=40_000, rng=5)
   print(f"Fourier: {prices[6]:.4f}, Monte Carlo: {mc.estimate:.4f} +/- {mc.std_error:.4f}")
   # Fourier: 5.9237, Monte Carlo: 5.9039 +/- 0.0339

3. The distribution behind the prices
-------------------------------------

Breeden and Litzenberger showed that the second derivative of call prices
in the strike is the discounted risk-neutral density. Differencing a fine
strip of Heston prices recovers it:

.. plot::
   :context: close-figs

   from scipy.integrate import trapezoid

   strikes = np.linspace(40.0, 200.0, 801)
   density = fk.options.breeden_litzenberger_density(strikes, fk.stochastic_volatility.heston_price(heston, S, strikes, T, r), T, r)
   x, p = density.strikes, density.density
   mean = trapezoid(x * p, x)
   sd = np.sqrt(trapezoid((x - mean) ** 2 * p, x))
   print(f"total probability {trapezoid(p, x):.4f}")
   # total probability 0.9994
   print(f"mean {mean:.2f} against the forward {S * np.exp(r * T):.2f}")
   # mean 101.49 against the forward 101.51
   print(f"skewness {trapezoid((x - mean) ** 3 * p, x) / sd**3:.2f}")
   # skewness -0.89

The density integrates to one and is centred on the forward, as the
absence of arbitrage requires. Its negative skewness is the skew: a fat
left tail makes low-strike puts expensive, so their implied volatility
is high. A lognormal with the same volatility would have positive
skewness.

4. Jumps smile too
------------------

Stochastic volatility is not the only explanation. Merton added
occasional jumps to the stock, and downward jumps also fatten the left
tail:

.. plot::
   :context: close-figs

   merton = fk.stochastic_volatility.MertonJumpModel(sigma=0.15, lam=0.5, mu_j=-0.1, delta=0.1)
   print(np.round(100 * merton.implied_vol(S, K[::3], T, r), 2))
   # [22.5  19.5  17.44 16.6  16.33]

The two models leave different fingerprints across maturities, though.
Jumps make the smile steep for short expiries and flat for long ones,
while stochastic volatility does the opposite, which is why the Bates
model combines them.

5. What traders quote: SABR
---------------------------

Desks rarely quote a full model. They fit a smile parametrization, and
Hagan's SABR formula is the standard in rates and FX. Calibrating it to
the Heston smile above:

.. plot::
   :context: close-figs

   forward = S * np.exp(r * T)
   sabr = fk.stochastic_volatility.sabr_calibrate(forward, K, T, smile.vols, beta=0.5)
   print(f"rho {sabr.rho:.3f}, nu {sabr.nu:.3f}, RMSE {1e4 * sabr.rmse:.0f} bp of vol")
   # rho -0.646, nu 0.824, RMSE 35 bp of vol

SABR's correlation ``rho`` and vol of vol ``nu`` play the same roles as
Heston's ``rho`` and ``xi``, and the fit is close: three parameters
summarise the whole smile.

See Also
--------

- :doc:`/api/options`
- :doc:`/api/stochastic_volatility`
- :doc:`/history/options_breakthroughs`
- :doc:`/history/stochastic_volatility_breakthroughs`
