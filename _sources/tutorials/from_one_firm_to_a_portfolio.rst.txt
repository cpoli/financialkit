:orphan:

Credit Risk, from One Firm to a Portfolio
=========================================

Credit risk has two questions: how likely is one borrower to default, and
how do defaults in a portfolio move together? This tutorial answers both
with :mod:`financialkit.credit`, which leans on
:mod:`financialkit.options` for its structural models. It starts with
one firm's balance sheet, moves to the hazard rates that credit markets
quote, and ends with the loss distribution, and the regulatory capital,
of a whole loan book.

1. Equity is a call option on the firm
--------------------------------------

Merton's insight was that shareholders hold a call option on the firm's
assets, struck at the face value of its debt: if the assets are worth
less than the debt when it falls due, the shareholders walk away. So
Black-Scholes prices equity, and debt is what remains:

.. code-block:: python

   import numpy as np
   import financialkit as fk

   firm = fk.credit.merton_model(V=150.0, D=100.0, T=1.0, r=0.03, sigma_V=0.20)
   print(f"equity {firm.equity:.2f}, the Black-Scholes call: {fk.options.black_scholes_price(150.0, 100.0, 1.0, 0.03, 0.20):.2f}")
   # equity 53.08, the Black-Scholes call: 53.08
   print(f"debt {firm.debt:.2f}, credit spread {1e4 * firm.credit_spread:.1f} bp")
   # debt 96.92, credit spread 12.9 bp
   print(f"risk-neutral default probability {100 * firm.default_probability:.2f}%, distance to default {firm.distance_to_default:.2f}")
   # risk-neutral default probability 1.89%, distance to default 2.08

The firm is 1.5 times covered and its spread is only 13 basis points.
That is Merton's well-known weakness: with continuous asset paths a firm
far from its default point can hardly default within a year, so
short-dated spreads come out far below market levels.

2. The KMV inversion: from what you can see
-------------------------------------------

Asset values are not observable, but equity prices and their volatility
are. Moody's KMV inverts Merton's two equations (equity as a call, and
equity volatility as asset volatility levered up by the call's delta) to
recover them:

.. code-block:: python

   implied = fk.credit.merton_calibrate(E=firm.equity, sigma_E=firm.equity_volatility, D=100.0, T=1.0, r=0.03)
   print(f"equity volatility {100 * firm.equity_volatility:.1f}% implies assets {implied.asset_value:.2f} with volatility {100 * implied.asset_volatility:.1f}%")
   # equity volatility 55.9% implies assets 150.00 with volatility 20.0%
   print(f"distance to default with an 8% real-world drift: {fk.credit.distance_to_default(150.0, 100.0, 0.08, 0.20):.2f}")
   # distance to default with an 8% real-world drift: 2.33

Equity is almost three times as volatile as the assets, because the
shareholders bear all the firm's risk on a thinner base. KMV then maps
the distance to default to a default frequency using historical default
data, not the normal distribution.

3. Hazard rates: how credit markets quote
-----------------------------------------

Reduced-form models skip the balance sheet and model default as the
first jump of a Poisson process with intensity :math:`\lambda`, the
hazard rate. A flat 2% hazard rate gives survival :math:`e^{-\lambda t}`,
and the credit default swap that insures against it has a par spread
close to :math:`\lambda (1 - R)`, the "credit triangle":

.. code-block:: python

   curve = fk.credit.HazardCurve.flat(0.02)
   print(f"5-year default probability {100 * curve.default_probability(5.0):.2f}%")
   # 5-year default probability 9.52%
   spread = fk.credit.cds_par_spread(curve, 5.0, r=0.03, recovery=0.4)
   print(f"5-year CDS par spread {1e4 * spread:.1f} bp; credit triangle {1e4 * fk.credit.spread_from_hazard(0.02, 0.4):.0f} bp")
   # 5-year CDS par spread 120.5 bp; credit triangle 120 bp

In practice the direction is reversed: CDS quotes at several maturities
are bootstrapped into a hazard curve, as
:func:`~financialkit.credit.bootstrap_hazard_curve` does.

4. A portfolio: correlation is everything
-----------------------------------------

Lend to 100 firms, each with a 2% default probability and a 45% loss
given default. In Vasicek's one-factor model, each firm's assets load on
a common factor with correlation :math:`\rho`. The expected loss does
not depend on :math:`\rho` at all, but the tail does:

.. code-block:: python

   for rho in (0.0, 0.15, 0.30):
       book = fk.credit.one_factor_loss_distribution(0.02, rho, 100, lgd=0.45)
       print(f"rho = {rho:.2f}: expected loss {100 * book.mean:.2f}%, 99.9% loss quantile {100 * book.quantile(0.999):.2f}%")
   # rho = 0.00: expected loss 0.90%, 99.9% loss quantile 3.15%
   # rho = 0.15: expected loss 0.90%, 99.9% loss quantile 9.00%
   # rho = 0.30: expected loss 0.90%, 99.9% loss quantile 15.75%

Without correlation, a one-in-a-thousand-year loss is three and a half times the
expected loss. With 30% asset correlation it is more than seventeen
times: defaults arrive together, in recessions. Simulating correlated
default times with Li's Gaussian copula shows the same thing name by name:

.. code-block:: python

   for rho in (0.0, 0.30):
       times = fk.credit.copula_default_times(curve, rho, 20_000, n_names=100, rng=1)
       defaults = (times < 5.0).sum(axis=1)
       print(f"rho = {rho:.2f}: mean 5-year defaults {defaults.mean():.1f}, 99.9% quantile {np.quantile(defaults, 0.999):.0f}")
   # rho = 0.00: mean 5-year defaults 9.5, 99.9% quantile 19
   # rho = 0.30: mean 5-year defaults 9.5, 99.9% quantile 73

5. Regulatory capital: the Basel formula
----------------------------------------

As the number of loans grows, the loss distribution converges to
Vasicek's closed form, and Basel II turned its 99.9% quantile into the
capital banks must hold against unexpected loss, with a correlation that
falls as the default probability rises:

.. code-block:: python

   rho_basel = fk.credit.basel_correlation(0.02)
   k = fk.credit.basel_irb_capital(0.02, lgd=0.45, maturity=1.0)
   by_hand = 0.45 * (fk.credit.vasicek_quantile(0.999, 0.02, rho_basel) - 0.02)
   print(f"Basel correlation {rho_basel:.3f}")
   # Basel correlation 0.164
   print(f"capital {100 * k:.2f}% of exposure; LGD x (99.9% Vasicek quantile - PD) = {100 * by_hand:.2f}%")
   # capital 7.66% of exposure; LGD x (99.9% Vasicek quantile - PD) = 7.66%

At a one-year maturity the Basel maturity adjustment is one, so the
capital charge is exactly the unexpected loss of an infinitely granular
Vasicek portfolio: the expected loss is priced into the loan, and capital
covers the rest of the tail.

See Also
--------

- :doc:`/api/credit`
- :doc:`/api/options`
- :doc:`/history/credit_breakthroughs`
