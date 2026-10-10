Breakthroughs in Credit Risk
============================


.. include:: /_generated/nav/credit.rst

.. epigraph::

   "Credit is a system whereby a person who can't pay gets another
   person who can't pay to guarantee that he can pay."
   -- Charles Dickens, *Little Dorrit* (1857)

For most of the twentieth century credit analysis meant reading
financial statements and assigning a letter grade. In under four
decades it became quantitative: statistics on accounting ratios, option
pricing applied to the firm, Poisson processes for the timing of
default, Markov chains for ratings, and factor models and copulas for
whole portfolios, culminating in the formula that sets bank capital
worldwide and the CDS conventions that still govern the market. This
chronology traces the ideas behind :mod:`financialkit.credit`.

.. contents:: Timeline
   :local:
   :depth: 1

1968 -- Altman's Z-Score
------------------------

Edward Altman applied Fisher's linear discriminant analysis to 22
accounting ratios of 66 US manufacturers, half of which had gone
bankrupt, and kept the five that separated them best: liquidity,
accumulated profitability, operating profitability, market leverage and
asset turnover. The weighted sum classified 95% of his sample correctly
a year before failure, and its thresholds of 1.81 and 2.99 are still
quoted.

.. math::

   Z = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 1.0 X_5

*Implementation:* :func:`financialkit.credit.systems.scoring.altman_z_score`,
:func:`~financialkit.credit.systems.scoring.altman_zone` and
:func:`~financialkit.credit.systems.scoring.linear_discriminant`. The
example scores synthetic healthy and failing firms and re-estimates the
discriminant direction.

*References:* E. I. Altman, "Financial Ratios, Discriminant Analysis and
the Prediction of Corporate Bankruptcy," *J. Finance* 23 (1968) 589-609.

.. minigallery:: ../../examples/credit/scoring/plot_01_altman_z_score.py

1974 -- Merton's Structural Model
---------------------------------

Robert Merton observed that the shareholders of a firm with zero-coupon
debt hold a call option on its assets: at maturity they repay the debt
if the assets are worth more, and hand over the firm otherwise. The
Black-Scholes formula then values equity, debt and the credit spread
from leverage and asset volatility alone, and the risk-neutral default
probability is :math:`N(-d_2)`.

.. math::

   E = V N(d_1) - D e^{-rT} N(d_2), \qquad D_{\text{market}} = V - E

*Implementation:* :func:`financialkit.credit.systems.structural.merton_model`
and :func:`~financialkit.credit.systems.structural.merton_calibrate`,
which recovers asset value and volatility from equity and its
volatility (Hull's Example 24.3). The example plots the payoffs and the
term structure of Merton spreads.

*References:* R. C. Merton, "On the Pricing of Corporate Debt: The Risk
Structure of Interest Rates," *J. Finance* 29 (1974) 449-470.

.. minigallery:: ../../examples/credit/structural/plot_01_merton_structural.py

1976 -- Black and Cox: First Passage
------------------------------------

Fischer Black and John Cox let bond covenants trigger default the first
time the assets fall to a barrier, at any date before maturity. Via the
reflection principle for Brownian motion the default probability is
again in closed form, and it is always at least the probability of
ending below the barrier that Merton's model uses.

.. math::

   Q(t) = N\left(\frac{h + m t}{\sigma\sqrt t}\right) - e^{-2 m h/\sigma^2} N\left(\frac{-h + m t}{\sigma\sqrt t}\right)

*Implementation:* :func:`financialkit.credit.systems.first_passage.black_cox_survival`,
:func:`~financialkit.credit.systems.first_passage.black_cox_default_probability`
and the Brownian-bridge Monte Carlo check
:func:`~financialkit.credit.systems.first_passage.first_passage_crossing_probability`.
The example compares the closed form with simulated asset paths and with
Merton.

*References:* F. Black and J. C. Cox, "Valuing Corporate Securities:
Some Effects of Bond Indenture Provisions," *J. Finance* 31 (1976)
351-367.

.. minigallery:: ../../examples/credit/structural/plot_02_black_cox.py

1987 -- Vasicek's Large Homogeneous Portfolio
---------------------------------------------

Oldrich Vasicek, at KMV, put Merton's model into a portfolio. When every
loan's asset return loads on one common factor, defaults are
independent given the factor, and in a large pool the default rate
converges to a deterministic function of it. Its distribution has a
closed form whose quantiles give the capital a lender needs, and the
same formula sits inside the Basel II rules.

.. math::

   P(L \le x) = N\left(\frac{\sqrt{1-\rho}\, N^{-1}(x) - N^{-1}(p)}{\sqrt\rho}\right)

*Implementation:* :func:`financialkit.credit.systems.vasicek.vasicek_cdf`,
:func:`~financialkit.credit.systems.vasicek.vasicek_pdf`,
:func:`~financialkit.credit.systems.vasicek.vasicek_quantile` and
:func:`~financialkit.credit.systems.vasicek.simulate_one_factor_defaults`.
The example shows the density for several correlations and finite pools
converging to the limit.

*References:* O. Vasicek, "Probability of Loss on Loan Portfolio" (KMV,
1987); O. Vasicek, "The Distribution of Loan Portfolio Value," *Risk* 15
(2002) 160-162.

.. minigallery:: ../../examples/credit/portfolio_loss/plot_01_vasicek_lhp.py

1994 -- Leland's Endogenous Default
-----------------------------------

Hayne Leland priced perpetual debt when coupons are tax deductible and
bankruptcy destroys part of the firm. Shareholders choose when to stop
paying, so the default barrier is set by smooth pasting rather than by
a covenant. Firm value trades the tax shield against expected
bankruptcy costs, and its maximum gives an optimal leverage, about 75%
with a 75 bp spread in Leland's base case.

.. math::

   V_B = \frac{(1 - \tau) C}{r + \sigma^2/2}

*Implementation:* :func:`financialkit.credit.systems.leland.leland_model`
and :func:`~financialkit.credit.systems.leland.leland_optimal_coupon`.
The example shows smooth pasting at the endogenous barrier and the
capital-structure trade-off.

*References:* H. E. Leland, "Corporate Debt Value, Bond Covenants, and
Optimal Capital Structure," *J. Finance* 49 (1994) 1213-1252.

.. minigallery:: ../../examples/credit/structural/plot_03_leland.py

1995 -- Jarrow and Turnbull: Hazard Rates
-----------------------------------------

Robert Jarrow and Stuart Turnbull stopped asking why firms default and
modelled only when: default is the first jump of a Poisson process.
Survival probabilities are exponentials of the integrated intensity,
risky bonds are priced like riskless ones with a survival factor, and
spreads, hazard rates and recovery are linked by the credit triangle.
These reduced-form models became the language of credit derivatives.

.. math::

   Q(t) = e^{-\int_0^t \lambda(s)\,ds}, \qquad P^d(0, T) = P(0, T)[\delta + (1 - \delta) Q(T)]

*Implementation:* :class:`financialkit.credit.systems.hazard.HazardCurve`
(piecewise-constant intensity, survival, density and sampling of
default times) and
:func:`~financialkit.credit.systems.hazard.jarrow_turnbull_zero`. The
example simulates default times and compares bond spreads with the
credit triangle.

*References:* R. A. Jarrow and S. M. Turnbull, "Pricing Derivatives on
Financial Securities Subject to Credit Risk," *J. Finance* 50 (1995)
53-85.

.. minigallery:: ../../examples/credit/reduced_form/plot_01_jarrow_turnbull.py

1997 -- CreditMetrics
---------------------

J. P. Morgan's CreditMetrics did for credit what RiskMetrics had done
for market risk. A bond is revalued in every rating it could migrate to
over a year, the values are weighted with a rating agency's transition
matrix, and credit VaR is read from the distribution. Joint migrations
come from correlated asset returns mapped to ratings by normal
thresholds. The Technical Document's five-year BBB bond has mean value
107.09 and standard deviation 2.99.

*Implementation:* :func:`financialkit.credit.systems.migration.creditmetrics_bond`,
:func:`~financialkit.credit.systems.migration.migration_thresholds`,
:func:`~financialkit.credit.systems.migration.joint_migration_probabilities`
and :func:`~financialkit.credit.systems.migration.creditmetrics_portfolio`,
with the published matrix in
:func:`~financialkit.credit.utils.matrices.creditmetrics_matrix`. The
example reproduces the BBB bond and simulates a correlated portfolio.

*References:* J. P. Morgan, *CreditMetrics -- Technical Document* (1997).

.. minigallery:: ../../examples/credit/migration/plot_01_creditmetrics.py

1997 -- Jarrow, Lando and Turnbull: Generator Matrices
------------------------------------------------------

Robert Jarrow, David Lando and Stuart Turnbull moved rating migration
to continuous time. A generator matrix of migration intensities gives
the transition matrix for any horizon through the matrix exponential,
so default probabilities and credit spreads exist for every maturity
and every rating, and a risk-premium adjustment links historical and
pricing measures.

.. math::

   P(t) = e^{\Lambda t}

*Implementation:* :func:`financialkit.credit.systems.migration.transition_from_generator`
(:func:`scipy.linalg.expm`) and
:func:`~financialkit.credit.systems.migration.jlt_risky_zero`. The
example plots default-probability term structures and spreads by rating.

*References:* R. A. Jarrow, D. Lando and S. M. Turnbull, "A Markov Model
for the Term Structure of Credit Risk Spreads," *Rev. Financial Stud.*
10 (1997) 481-523.

.. minigallery:: ../../examples/credit/migration/plot_02_generator_matrix.py

1997 -- CreditRisk+ and Panjer's Recursion
------------------------------------------

Credit Suisse First Boston brought actuarial science to loan portfolios.
Defaults are Poisson events whose rates share a gamma-distributed sector
factor, exposures are banded into integer loss units, and the loss
distribution follows exactly from Panjer's 1981 recursion for compound
distributions, with no simulation at all.

.. math::

   f_n = \frac{1}{1 - a s_0} \sum_{j=1}^{n} \left(a + \frac{b j}{n}\right) s_j f_{n-j}

*Implementation:* :func:`financialkit.credit.systems.creditrisk_plus.creditrisk_plus`
on top of :func:`~financialkit.credit.utils.numerics.panjer_recursion`.
The example shows how sector volatility fattens the tail.

*References:* Credit Suisse First Boston, *CreditRisk+: A Credit Risk
Management Framework* (1997); H. H. Panjer, "Recursive Evaluation of a
Family of Compound Distributions," *ASTIN Bull.* 12 (1981) 22-26.

.. minigallery:: ../../examples/credit/portfolio_loss/plot_02_creditrisk_plus.py

1999 -- Duffie and Singleton: Recovery of Market Value
------------------------------------------------------

Darrell Duffie and Kenneth Singleton assumed that a defaulting claim
loses a fraction of its own market value. Defaultable claims are then
priced exactly like riskless ones under the default-adjusted short rate
:math:`r + \lambda L`, so affine term-structure models price credit too,
including stochastic intensities.

.. math::

   P^d(0, T) = E\left[e^{-\int_0^T (r_s + \lambda_s L)\,ds}\right]

*Implementation:* :func:`financialkit.credit.systems.recovery.risky_zero_coupon`
(market value, treasury and face-value recovery) and
:func:`~financialkit.credit.systems.recovery.duffie_singleton_cir_zero`.
The example compares the conventions and checks the CIR closed form by
simulation.

*References:* D. Duffie and K. J. Singleton, "Modeling Term Structures
of Defaultable Bonds," *Rev. Financial Stud.* 12 (1999) 687-720.

.. minigallery:: ../../examples/credit/reduced_form/plot_02_duffie_singleton.py

2000 -- Li's Gaussian Copula
----------------------------

David Li joined default times with a Gaussian copula. Each name keeps
the survival curve implied by its CDS spreads, and one correlation
parameter controls how often names default together. The model priced
basket swaps and CDOs for a decade, and its thin joint tail made it a
symbol of the 2007-2008 crisis.

.. math::

   P(\tau_1 \le T, \tau_2 \le T) = \Phi_2\big(N^{-1}(p_1), N^{-1}(p_2); \rho\big)

*Implementation:* :func:`financialkit.credit.systems.copula_defaults.copula_default_times`,
:func:`~financialkit.credit.systems.copula_defaults.joint_default_probability`,
:func:`~financialkit.credit.systems.copula_defaults.default_correlation`
and :func:`~financialkit.credit.systems.copula_defaults.nth_to_default_probability`.
The example shows joint default times and nth-to-default probabilities.

*References:* D. X. Li, "On Default Correlation: A Copula Function
Approach," *J. Fixed Income* 9 (2000) 43-54.

.. minigallery:: ../../examples/credit/portfolio_loss/plot_03_li_gaussian_copula.py

2001 -- Israel, Rosenthal and Wei: Finding Generators
-----------------------------------------------------

Agencies publish annual transition matrices, but continuous-time models
need generators. Robert Israel, Jeffrey Rosenthal and Jason Wei showed
that the matrix logarithm of an empirical rating matrix usually has
negative off-diagonal entries, which would be negative migration
intensities, gave conditions under which no valid generator exists, and
proposed zeroing the offending entries and restoring the row sums.

*Implementation:* :func:`financialkit.credit.systems.migration.generator_from_transition`
(:func:`scipy.linalg.logm`),
:func:`~financialkit.credit.systems.migration.is_valid_generator` and
:func:`~financialkit.credit.systems.migration.regularize_generator`
(diagonal and weighted adjustments). The example regularizes the log of
the CreditMetrics matrix and measures the error.

*References:* R. B. Israel, J. S. Rosenthal and J. Z. Wei, "Finding
Generators for Markov Chains via Empirical Transition Matrices, with
Applications to Credit Ratings," *Math. Finance* 11 (2001) 245-265.

.. minigallery:: ../../examples/credit/migration/plot_03_israel_rosenthal_wei.py

2003 -- Crosbie and Bohn: Distance to Default
---------------------------------------------

Moody's KMV turned Merton's model into a commercial default predictor.
Peter Crosbie and Jeffrey Bohn described backing asset values out of a
daily equity series by iteration, a default point of short-term debt
plus half the long-term debt, and the distance to default, the number
of standard deviations separating the two, mapped to an expected
default frequency.

.. math::

   DD = \frac{\ln(V/D) + (\mu - \sigma_V^2/2) T}{\sigma_V \sqrt T}

*Implementation:* :func:`financialkit.credit.systems.structural.kmv_iterate`,
:func:`~financialkit.credit.systems.structural.kmv_default_point` and
:func:`~financialkit.credit.systems.structural.distance_to_default`.
The example runs the iteration and a rolling distance to default.

*References:* P. J. Crosbie and J. R. Bohn, *Modeling Default Risk*
(Moody's KMV, 2003).

.. minigallery:: ../../examples/credit/structural/plot_04_distance_to_default.py

2003 -- O'Kane and Turnbull: The CDS Par Spread
-----------------------------------------------

Dominic O'Kane and Stuart Turnbull set out the market-standard valuation
of credit default swaps: the premium leg as a risky annuity including
premium accrued at default, the protection leg as an integral over the
default density, both evaluated on a grid with defaults at mid-step.
The par spread equates them and reduces to :math:`\lambda(1 - R)` in
the continuous limit.

.. math::

   s = \frac{(1 - R)\int_0^T Z(u)\,(-dQ(u))}{\text{RPV01}}

*Implementation:* :func:`financialkit.credit.systems.cds.cds_legs` and
:func:`~financialkit.credit.systems.cds.cds_par_spread`. The example
works Hull's annual setting by hand and shows the approach to the
credit triangle.

*References:* D. O'Kane and S. Turnbull, *Valuation of Credit Default
Swaps* (Lehman Brothers, 2003).

.. minigallery:: ../../examples/credit/cds/plot_01_cds_par_spread.py

2004 -- The Basel II IRB Formula
--------------------------------

The Basel Committee's revised framework let banks use their own default
probabilities and set capital with Vasicek's model at 99.9% confidence,
less expected loss, times a maturity adjustment, with supervisory
correlations by asset class. Michael Gordy had shown that this is the
only setting in which a loan's charge does not depend on the rest of the
portfolio.

.. math::

   K = \left[LGD\, N\left(\frac{N^{-1}(PD) + \sqrt R N^{-1}(0.999)}{\sqrt{1 - R}}\right) - PD \cdot LGD\right] \frac{1 + (M - 2.5) b}{1 - 1.5 b}

*Implementation:* :func:`financialkit.credit.systems.basel.basel_irb_capital`,
:func:`~financialkit.credit.systems.basel.basel_risk_weight` and
:func:`~financialkit.credit.systems.basel.basel_correlation`. The
example plots risk weights by asset class and the quantile behind them.

*References:* Basel Committee on Banking Supervision, *International
Convergence of Capital Measurement and Capital Standards: A Revised
Framework* (2004); M. B. Gordy, *J. Financial Intermediation* 12 (2003)
199-232.

.. minigallery:: ../../examples/credit/portfolio_loss/plot_04_basel_irb.py

2004 -- Hull and White: Tranches Without Monte Carlo
----------------------------------------------------

John Hull and Alan White priced CDO tranches and nth-to-default swaps in
the one-factor copula semi-analytically: conditional on the factor,
defaults are independent and the pool loss is binomial, so expected
tranche losses at each payment date are a single integral. Correlation
moves risk between the equity and senior tranches, which is why tranche
quotes were read as implied correlations.

.. math::

   L_{[A, D]} = \frac{\min(L, D) - \min(L, A)}{D - A}

*Implementation:* :func:`financialkit.credit.systems.tranches.expected_tranche_loss`
and :func:`~financialkit.credit.systems.tranches.tranche_spread`. The
example shows tranche spreads against correlation and finite pools
against the large-pool limit.

*References:* J. Hull and A. White, "Valuation of a CDO and an n-th to
Default CDS Without Monte Carlo Simulation," *J. Derivatives* 12 (2004)
8-23.

.. minigallery:: ../../examples/credit/portfolio_loss/plot_05_tranche_expected_loss.py

2009 -- The ISDA Standard Model: Hazard Bootstrapping
-----------------------------------------------------

The 2009 "Big Bang" standardized CDS contracts: fixed coupons of 100 or
500 bp, an upfront payment, and an open-source ISDA standard model to
convert between upfronts and spreads. Its core is a bootstrap of a
piecewise-constant hazard curve, one maturity at a time, so that every
quoted CDS reprices exactly.

*Implementation:* :func:`financialkit.credit.systems.cds.bootstrap_hazard_curve`,
:func:`~financialkit.credit.systems.cds.cds_upfront` and
:func:`~financialkit.credit.systems.cds.upfront_to_spread` (flat rate,
even payment grid; ISDA day counts and IMM dates are not modelled). The
example bootstraps a term structure of quotes and computes upfronts at
both standard coupons.

*References:* ISDA, *CDS Standard Model* (2009); Markit, *The CDS Big
Bang* (2009).

.. minigallery:: ../../examples/credit/cds/plot_02_isda_hazard_bootstrap.py

See Also
--------

- :doc:`/api/credit`
- :doc:`/history/options_breakthroughs`
- :doc:`/history/interest_rate_models_breakthroughs`
- :doc:`/history/risk_breakthroughs`
