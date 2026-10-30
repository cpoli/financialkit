Breakthroughs in Option Pricing
===============================


.. include:: /_generated/nav/options.rst

.. epigraph::

   "The Black-Scholes formula is still around, even though it depends on
   at least 10 unrealistic assumptions."
   -- attributed to Fischer Black

An option gives its holder the right, but not the obligation, to buy or
sell at a fixed price. Options traded in Amsterdam in the seventeenth
century and in London and Paris long before anyone could say what one
was worth. The answer came in stages: a model of random prices in 1900,
model-free relations in the late 1960s, the Black-Scholes replication
argument in 1973, then the trees, approximations and inversions that
turned one formula into a trading tool, the smile that showed where the
formula fails, and the cash dividends, FX quotes, machine-precision
inversion and negative prices that practice demanded since. This
chronology traces the ideas behind :mod:`financialkit.options`.

.. contents:: Timeline
   :local:
   :depth: 1

1900 -- Bachelier's Theory of Speculation
-----------------------------------------

Louis Bachelier's doctoral thesis, defended at the Sorbonne under Henri
Poincaré, modelled the price of French government rentes as a random
walk whose continuous limit is what we now call Brownian motion, five
years before Einstein's paper on the subject. He derived the normal
distribution of price changes, its spread growing with the square root
of time, and priced the options traded on the Paris Bourse. The thesis
was largely forgotten until Leonard Savage and Paul Samuelson
rediscovered it in the 1950s. Its one flaw for stocks, that prices can
go negative, is harmless for rates and spreads, where the normal model
is still quoted today.

.. math::

   c = e^{-rT}\left[(F - K) N(d) + \sigma_N\sqrt{T}\, n(d)\right],
   \qquad d = \frac{F - K}{\sigma_N\sqrt{T}}

*Implementation:* :func:`financialkit.options.systems.closed_form.bachelier_price`
prices calls and puts in the normal model, with an absolute volatility.
The example checks it against simulated arithmetic Brownian paths and
shows the square-root-of-time law.

*References:* L. Bachelier, "Théorie de la spéculation," *Annales
scientifiques de l'École Normale Supérieure* 17 (1900) 21-86.

.. minigallery:: ../../examples/options/closed_form/plot_01_bachelier.py

1969 -- Stoll's Put-Call Parity
-------------------------------

Hans Stoll observed that a long call and a short put with the same
strike and maturity pay :math:`S_T - K` in every state, which is a
forward contract. Their prices must therefore differ by the value of
that forward, whatever model drives the stock. The relation was known
to traders, who used conversions and reversals to turn calls into puts,
but Stoll's paper made it the first piece of option theory that needs
no assumption about the distribution of prices.

.. math::

   c - p = S e^{-qT} - K e^{-rT}

*Implementation:* :func:`financialkit.options.systems.parity.put_call_parity`
gives one leg from the other, and
:func:`~financialkit.options.systems.parity.parity_residual` measures the
arbitrage profit in a mispriced pair. The example shows the residual
vanishing for Black-Scholes prices and the riskless payoff of a reversal.

*References:* H. R. Stoll, "The Relationship Between Put and Call Option
Prices," *Journal of Finance* 24 (1969) 801-824.

.. minigallery:: ../../examples/options/closed_form/plot_02_stoll_put_call_parity.py

1973 -- The Black-Scholes Formula
---------------------------------

Fischer Black and Myron Scholes assumed that the stock follows geometric
Brownian motion and that trading is continuous and frictionless. A
portfolio of the option and the right number of shares is then
instantaneously riskless, so it must earn the risk-free rate. That
yields a partial differential equation whose solution for a European
call does not involve the stock's expected return. The paper was
rejected twice before the *Journal of Political Economy* published it,
weeks after the Chicago Board Options Exchange opened. Scholes and
Robert Merton received the 1997 Nobel prize in economics; Black had died
in 1995.

.. math::

   c = S N(d_1) - K e^{-rT} N(d_2), \qquad
   d_{1,2} = \frac{\ln(S/K) + (r \pm \sigma^2/2) T}{\sigma\sqrt{T}}

*Implementation:* :func:`financialkit.options.systems.closed_form.black_scholes_price`
broadcasts over arrays of strikes, maturities and spots. The example
checks it against a risk-neutral Monte Carlo over
:func:`financialkit.integrators.gbm_paths`.

*References:* F. Black and M. Scholes, "The Pricing of Options and
Corporate Liabilities," *Journal of Political Economy* 81 (1973)
637-654.

.. minigallery:: ../../examples/options/closed_form/plot_03_black_scholes.py

1973 -- Merton: Dividends and No-Arbitrage Bounds
-------------------------------------------------

Robert Merton's *Theory of Rational Option Pricing*, published the same
year, separated what follows from the absence of arbitrage alone from
what needs a model. Among the model-free results are bounds on every
European option price, and the fact that an American call on a
non-dividend-paying stock should never be exercised early. Merton then
rederived the Black-Scholes formula under weaker assumptions and
extended it to a continuous dividend yield, stochastic interest rates
and barrier options.

.. math::

   \max(S e^{-qT} - K e^{-rT}, 0) \le c \le S e^{-qT}, \qquad
   c = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)

*Implementation:* :func:`financialkit.options.systems.parity.no_arbitrage_bounds`
gives the bounds, and the ``q`` argument of
:func:`~financialkit.options.systems.closed_form.black_scholes_price`
is Merton's dividend yield. The example reproduces Hull's index option
and shows the prices filling the region between the bounds.

*References:* R. C. Merton, "Theory of Rational Option Pricing," *Bell
Journal of Economics and Management Science* 4 (1973) 141-183.

.. minigallery:: ../../examples/options/closed_form/plot_04_merton_bounds_and_dividends.py

1973 -- Delta Hedging and the Greeks
------------------------------------

The heart of the Black-Scholes argument is the hedge: hold
:math:`\Delta = \partial V/\partial S` shares against each option and
the position is riskless over the next instant. Traders soon had Greek
letters for the other partial derivatives: gamma for how quickly delta
changes, theta for time decay, vega for volatility, rho for rates. The
pricing equation ties them together, so time decay pays for gamma. In
practice the hedge is rebalanced discretely, and Boyle and Emanuel
(1980) showed that the hedging error then has a spread proportional to
:math:`1/\sqrt{n}` in the number of rebalances.

.. math::

   \Theta + \tfrac12 \sigma^2 S^2 \Gamma + (r - q) S \Delta = r V

*Implementation:* :func:`financialkit.options.systems.greeks.black_scholes_greeks`
returns all first-order Greeks plus vanna and volga, tested against
:func:`~financialkit.options.utils.common.finite_difference_greeks`,
which is built on :func:`~financialkit.options.utils.common.bump_greek`,
the Greek in any named input of any pricer.
:func:`financialkit.options.systems.hedging.delta_hedge` simulates a
short option hedged at a chosen frequency. The examples check the
Greeks and the PDE, then measure the hedging error.

*References:* F. Black and M. Scholes, *Journal of Political Economy* 81
(1973) 637-654; P. P. Boyle and D. Emanuel, "Discretely Adjusted Option
Hedges," *Journal of Financial Economics* 8 (1980) 259-282.

.. minigallery:: ../../examples/options/greeks/plot_01_greeks.py ../../examples/options/greeks/plot_02_delta_hedging.py

1976 -- Black's Model for Futures Options
-----------------------------------------

Fischer Black adapted the formula to commodity contracts. Entering a
futures position costs nothing, so the futures price has zero drift
under the risk-neutral measure, and the option formula keeps its shape
with the futures price in place of the spot. Because it needs only a
forward price and a discount factor, Black-76 became the market's
quoting convention for options on bonds, caps, floors and swaptions as
well.

.. math::

   c = e^{-rT}\left[F N(d_1) - K N(d_2)\right], \qquad
   d_{1,2} = \frac{\ln(F/K) \pm \sigma^2 T/2}{\sigma\sqrt{T}}

*Implementation:* :func:`financialkit.options.systems.closed_form.black76_price`
is the only Black-76 implementation in financialkit; the
Black-Scholes-Merton and Garman-Kohlhagen pricers call it on the
forward. The example reproduces Hull's futures put and the equivalence
with Black-Scholes.

*References:* F. Black, "The Pricing of Commodity Contracts," *Journal
of Financial Economics* 3 (1976) 167-179.

.. minigallery:: ../../examples/options/closed_form/plot_05_black76.py

1976 -- Latané and Rendleman's Implied Volatility
-------------------------------------------------

Every input to the Black-Scholes formula can be observed except the
volatility. Henry Latané and Richard Rendleman inverted the formula,
solving for the volatility that makes the model price equal the market
price, and found that these implied standard deviations predicted
future volatility better than historical estimates. Quoting options by
implied volatility instead of price soon became standard, because it
makes options with different strikes and maturities comparable.

.. math::

   \sigma_{\text{imp}}: \quad C_{\text{BS}}(S, K, T, r, \sigma_{\text{imp}}) = C_{\text{market}}

*Implementation:* :func:`financialkit.options.systems.implied_vol.implied_volatility`
checks Merton's bounds, then brackets the unique root for
:func:`scipy.optimize.brentq`. The example shows the monotone price
curve and the error raised for an arbitrageable price.

*References:* H. A. Latané and R. J. Rendleman, "Standard Deviations of
Stock Price Ratios Implied in Option Prices," *Journal of Finance* 31
(1976) 369-381.

.. minigallery:: ../../examples/options/implied_vol/plot_01_implied_volatility.py

1978 -- Breeden-Litzenberger: State Prices from Options
-------------------------------------------------------

Douglas Breeden and Robert Litzenberger noticed that a butterfly spread
of calls around a strike pays off only if the stock ends near that
strike. In the limit, its price is the Arrow-Debreu price of that
state, so the second derivative of the call price in the strike is the
discounted risk-neutral density. Option prices thus reveal the market's
entire distribution for the terminal price, not just a volatility.

.. math::

   f(K) = e^{rT} \frac{\partial^2 c}{\partial K^2}

*Implementation:* :func:`financialkit.options.systems.density.breeden_litzenberger_density`
takes the three-point second difference on any strictly increasing
strike grid. The example recovers the lognormal from Black-Scholes
prices and a bimodal density from a two-regime market.

*References:* D. T. Breeden and R. H. Litzenberger, "Prices of
State-Contingent Claims Implicit in Option Prices," *Journal of
Business* 51 (1978) 621-651.

.. minigallery:: ../../examples/options/implied_vol/plot_02_breeden_litzenberger.py

1979 -- The Cox-Ross-Rubinstein Binomial Tree
---------------------------------------------

John Cox, Stephen Ross and Mark Rubinstein replaced continuous trading
with a stock that moves up or down by a fixed factor in each period.
One period at a time, a call is replicated by shares and a bond, which
gives risk-neutral pricing with elementary algebra, and the
Black-Scholes formula reappears as the number of periods grows. The
tree did more than simplify the teaching: backward induction checks for
early exercise at every node, so it prices American options, which no
closed form did.

.. math::

   u = e^{\sigma\sqrt{\Delta t}} = 1/d, \qquad
   p = \frac{e^{(r - q)\Delta t} - d}{u - d}, \qquad
   V_i = e^{-r\Delta t}\left[p V_{i+1}^{u} + (1 - p) V_{i+1}^{d}\right]

*Implementation:* :class:`financialkit.options.systems.trees.CRRTree`
values European and American options and records the early-exercise
boundary. The examples show the :math:`1/n` convergence to
Black-Scholes, reproduce Hull's five-step American put, and plot the
put's exercise boundary.

*References:* J. C. Cox, S. A. Ross and M. Rubinstein, "Option Pricing:
A Simplified Approach," *Journal of Financial Economics* 7 (1979)
229-263.

.. minigallery:: ../../examples/options/trees/plot_01_crr_convergence.py ../../examples/options/trees/plot_02_crr_american_put.py

1979 -- The Rendleman-Bartter Tree
----------------------------------

Richard Rendleman and Brit Bartter published a two-state option model
independently, in the same year. They centred each step on the drift
of the log price rather than on today's price, so the up and down moves
are nearly equally likely and the lattice drifts with the stock. Both
trees match the first two moments of the log return in the limit and
converge to Black-Scholes at the same rate; the two parametrizations
are the prototypes of the many binomial schemes that followed.

.. math::

   \ln u, \ln d = (r - q - \tfrac12\sigma^2)\Delta t \pm \sigma\sqrt{\Delta t}

*Implementation:* :class:`financialkit.options.systems.trees.RendlemanBartterTree`
shares the backward induction of the other binomial trees. The example
compares its probabilities and node layout with CRR's and plots both
convergence curves.

*References:* R. J. Rendleman and B. J. Bartter, "Two-State Option
Pricing," *Journal of Finance* 34 (1979) 1093-1110.

.. minigallery:: ../../examples/options/trees/plot_03_rendleman_bartter.py

1983 -- Garman-Kohlhagen Currency Options
-----------------------------------------

Exchange-traded currency options began on the Philadelphia Stock
Exchange in December 1982. Mark Garman and Steven Kohlhagen priced them
by treating a unit of foreign currency as an asset that pays the
foreign interest rate continuously, just as Merton's stock pays a
dividend yield. The formula also makes the symmetry of FX options
explicit: a call on one currency is a put on the other.

.. math::

   c = S e^{-r_f T} N(d_1) - K e^{-r_d T} N(d_2)

*Implementation:* :func:`financialkit.options.systems.closed_form.garman_kohlhagen_price`
is Black-Scholes-Merton with ``q = r_f``. The example reproduces Haug's
currency call and checks the foreign-domestic symmetry.

*References:* M. B. Garman and S. W. Kohlhagen, "Foreign Currency Option
Values," *Journal of International Money and Finance* 2 (1983)
231-237.

.. minigallery:: ../../examples/options/closed_form/plot_06_garman_kohlhagen.py

1986 -- Boyle's Trinomial Tree
------------------------------

Phelim Boyle, who had introduced Monte Carlo option pricing in 1977,
added a third, unchanged branch to each step of the lattice. With
three probabilities he could match the mean and variance of the next
price exactly at every step, and a stretch parameter :math:`\lambda`
controlled how far apart the nodes were. The trinomial tree typically
reaches a given accuracy with fewer steps, and it is the lattice that
explicit finite-difference schemes for the Black-Scholes equation turn
out to be.

.. math::

   u = e^{\lambda\sigma\sqrt{\Delta t}}, \qquad
   p_u u + p_m + p_d u^{-1} = e^{(r - q)\Delta t}

*Implementation:* :class:`financialkit.options.systems.trees.TrinomialTree`
uses Boyle's moment-matching probabilities. The example verifies the
moments, compares convergence with CRR and varies the stretch.

*References:* P. P. Boyle, "Option Valuation Using a Three-Jump
Process," *International Options Journal* 3 (1986) 7-12.

.. minigallery:: ../../examples/options/trees/plot_04_boyle_trinomial.py

1987 -- The Barone-Adesi-Whaley Approximation
---------------------------------------------

Trees price American options but are slow when a trader needs
thousands of prices. Giovanni Barone-Adesi and Robert Whaley, following
Lawrence MacMillan (1986), noted that the early-exercise premium itself
solves the Black-Scholes equation. Neglecting one small term turns it
into an ordinary differential equation with power-law solutions, and
smooth pasting at a critical stock price fixes the remaining constant.
The result needs a single scalar root search and is accurate to a few
cents.

.. math::

   P = p + A_1 \left(\frac{S}{S^{**}}\right)^{q_1} \quad (S > S^{**}), \qquad
   P = K - S \quad (S \le S^{**})

*Implementation:* :func:`financialkit.options.systems.american.barone_adesi_whaley`
finds the critical price with :func:`scipy.optimize.brentq` and returns
it with the American and European values. The example compares speed
and accuracy with a 2000-step tree.

*References:* G. Barone-Adesi and R. E. Whaley, "Efficient Analytic
Approximation of American Option Values," *Journal of Finance* 42
(1987) 301-320; L. W. MacMillan, "Analytic Approximation for the
American Put Option," *Advances in Futures and Options Research* 1
(1986) 119-139.

.. minigallery:: ../../examples/options/trees/plot_05_barone_adesi_whaley.py

1987 -- The Volatility Smile
----------------------------

Under Black-Scholes every option on the same stock and maturity has
the same implied volatility. Before October 1987 index options came
close to that. After the crash of 19 October, when the S&P 500 fell
by about 20% in a day, out-of-the-money puts became permanently
expensive, and implied volatility has fallen steeply with the strike
ever since. Mark Rubinstein documented the change and built implied
trees to fit it. The smile is the market's correction to the lognormal
assumption: fat tails and a crash premium.

.. math::

   K \mapsto \sigma_{\text{imp}}(K, T) \neq \text{constant}

*Implementation:* :func:`financialkit.options.systems.implied_vol.implied_vol_smile`
inverts a strip of prices into a
:class:`~financialkit.options.core.base.SmileResult`. The example shows
a flat smile from Black-Scholes prices, a symmetric smile from
uncertain volatility, and a skew from a small probability of a crash.

*References:* M. Rubinstein, "Implied Binomial Trees," *Journal of
Finance* 49 (1994) 771-818.

.. minigallery:: ../../examples/options/implied_vol/plot_03_volatility_smile.py

1988 -- Brenner and Subrahmanyam's At-the-Money Formula
-------------------------------------------------------

Menachem Brenner and Marti Subrahmanyam showed that for an option
struck at the forward the Black-Scholes formula is almost linear in
volatility. A first-order expansion gives an implied volatility from a
single division, close enough for a trader on the floor and a good
starting point for an exact solver. The relative error is about
:math:`-\sigma^2 T / 24`, so it is tiny for short, at-the-money
options and grows away from the money.

.. math::

   \sigma \approx \sqrt{\frac{2\pi}{T}}\, \frac{c}{S e^{-qT}}

*Implementation:* :func:`financialkit.options.systems.implied_vol.brenner_subrahmanyam_vol`
applies the formula. The example compares it with the exact inversion
across volatilities, maturities and strikes.

*References:* M. Brenner and M. G. Subrahmanyam, "A Simple Formula to
Compute the Implied Standard Deviation," *Financial Analysts Journal*
44(5) (1988) 80-83.

.. minigallery:: ../../examples/options/implied_vol/plot_04_brenner_subrahmanyam.py

1993 -- The Bjerksund-Stensland American Approximation
------------------------------------------------------

Petter Bjerksund and Gunnar Stensland took a different route to the
American option. Instead of approximating the optimal policy's value,
they valued a simple policy exactly: exercise the first time the spot
touches a flat trigger. With a flat barrier, the option is a portfolio
of knock-out claims with closed forms. The trigger is placed between
the exercise boundary just before expiry and the perpetual boundary,
by a rule fitted to typical maturities. A feasible but suboptimal
policy can only be worth less than the optimal one, so the price is a
lower bound. It needs no root search, and for years it was the default
American pricer in trading systems.

.. math::

   C = \alpha S^\beta - \alpha\phi(S, T, \beta, I, I) + \phi(S, T, 1, I, I)
       - \phi(S, T, 1, K, I) - K\phi(S, T, 0, I, I) + K\phi(S, T, 0, K, I)

*Implementation:* :func:`financialkit.options.systems.american.bjerksund_stensland`
prices calls and, through the put-call transformation, puts, and returns
the trigger. The example reproduces Haug's worked value, shows the
price staying below a fine tree, and sets the flat trigger against the
tree's exercise boundary.

*References:* P. Bjerksund and G. Stensland, "Closed-Form Approximation
of American Options," *Scandinavian Journal of Management* 9 (1993)
S87-S99.

.. minigallery:: ../../examples/options/trees/plot_07_bjerksund_stensland.py

1996 -- The Leisen-Reimer Tree
------------------------------

The CRR price zigzags around Black-Scholes as the number of steps
grows, because the strike moves between nodes. Dietmar Leisen and
Matthias Reimer chose the tree's probabilities instead of its jumps:
they mapped Black-Scholes's :math:`d_1` and :math:`d_2` onto the
binomial distribution with the Peizer-Pratt normal approximation. The
tree is then centred on the strike, the oscillation disappears, and
European prices converge at order :math:`1/n^2`, so a 51-step tree is
more accurate than a CRR tree with thousands of steps.

.. math::

   p = h(d_2), \quad p' = h(d_1), \quad
   u = e^{(r - q)\Delta t} \frac{p'}{p}, \quad
   d = \frac{e^{(r - q)\Delta t} - p u}{1 - p}

*Implementation:* :class:`financialkit.options.systems.trees.LeisenReimerTree`
uses Peizer-Pratt method 2 and rounds the step count up to an odd
number. The example measures its second-order convergence and its
smooth convergence for American puts.

*References:* D. P. J. Leisen and M. Reimer, "Binomial Models for Option
Valuation -- Examining and Improving Convergence," *Applied Mathematical
Finance* 3 (1996) 319-346.

.. minigallery:: ../../examples/options/trees/plot_06_leisen_reimer.py

2003 -- Haug-Haug-Lewis: Discrete Cash Dividends
------------------------------------------------

Stocks pay fixed cash amounts on known dates, not a continuous yield.
The textbook fixes treat the dividends' present value as riskless and
subtract it from the spot (the escrowed model) or add it to the
strike, which changes the volatility of what is being modelled. The
errors are largest for dividends just before expiry, and trees with
cash dividends no longer recombine. Espen Haug, Jørgen Haug and Alan
Lewis went back to the definition. Just after a dividend, the option is
a plain European option on the reduced stock, so its value today is a
one-dimensional integral of the Black-Scholes price over the stock's
distribution just before the dividend, and several dividends nest the
integrals.

.. math::

   c_0 = e^{-r t_D}\int_0^\infty c_{BS}\big(\max(S - D, 0), K, T - t_D\big)\,
         f_{S_{t_D}}(S)\, dS

*Implementation:* :func:`financialkit.options.systems.dividends.discrete_dividend_price`
evaluates the nested integrals by Gauss-Legendre quadrature split at the
kinks of the integrand. The example checks it against a simulation and
against a dividend at expiry, then shows the escrowed model's error
growing as the dividend date approaches expiry.

*References:* E. G. Haug, J. Haug and A. Lewis, "Back to Basics: a New
Approach to the Discrete Dividend Problem," *Wilmott Magazine*
(September 2003).

.. minigallery:: ../../examples/options/closed_form/plot_08_discrete_dividends.py

2007 -- Castagna-Mercurio: the Vanna-Volga FX Smile
---------------------------------------------------

Currency option brokers quote three volatilities per expiry: at the
money, the 25-delta risk reversal and the 25-delta butterfly. FX
traders priced every other strike by a hedging rule of thumb. The
option is worth its Black-Scholes value at the at-the-money volatility,
plus the market cost of the three quoted options that hedge its vega,
vanna and volga. Antonio Castagna and Fabio Mercurio wrote the hedge
weights in closed form, proved that the rule reproduces the three
quotes, and derived approximations of the implied smile. The
second-order one is accurate far into the wings.

.. math::

   C(K) = C_{BS}(K, \sigma_{ATM}) + \sum_{i=1}^{3} \frac{\mathcal V(K)}{\mathcal V(K_i)}
          \prod_{j \ne i}\frac{\ln(K_j/K)}{\ln(K_j/K_i)}
          \big[C_{\text{mkt}}(K_i) - C_{BS}(K_i, \sigma_{ATM})\big]

*Implementation:* :func:`financialkit.options.systems.vanna_volga.vanna_volga_price`
and :func:`~financialkit.options.systems.vanna_volga.vanna_volga_smile`,
with the pillar strikes from
:func:`~financialkit.options.systems.vanna_volga.fx_pillar_strikes`. The
example builds a smile from broker quotes, rebuilds the price from the
vega-vanna-volga hedge, and compares the two approximations.

*References:* A. Castagna and F. Mercurio, "The Vanna-Volga Method for
Implied Volatilities," *Risk* 20(1) (2007) 106-111.

.. minigallery:: ../../examples/options/implied_vol/plot_06_vanna_volga.py

2015 -- Jäckel's "Let's Be Rational"
------------------------------------

Inverting Black's formula looks easy until the prices run to
:math:`10^{-80}` or come within rounding of their bounds. Then Newton's
method diverges, and bracketing solvers need a tolerance and dozens of
iterations. Peter Jäckel first normalised the problem to one price
function of two variables, log-moneyness and total volatility, and
evaluated it to full relative precision in every regime. He then
interpolated its inverse with rational cubics on four branches split at
the inflection point, and finished with Householder's third-order
method on a transformed objective for each branch. Two iterations reach
machine precision for every input. Open-source and commercial pricing
libraries have since adopted it.

.. math::

   b(x, s) = e^{x/2} N\!\left(\frac{x}{s} + \frac{s}{2}\right)
           - e^{-x/2} N\!\left(\frac{x}{s} - \frac{s}{2}\right), \qquad
   x = \ln\frac{F}{K}, \quad s = \sigma\sqrt{T}

*Implementation:* :mod:`financialkit.options.systems.lets_be_rational`
(:func:`~financialkit.options.systems.lets_be_rational.normalised_black_call`
and :func:`~financialkit.options.systems.lets_be_rational.normalised_implied_volatility`),
used by ``implied_volatility(..., method="jaeckel")`` and vectorised
over the whole array. The example draws the four branches, measures the
error after zero, one and two iterations, and compares the result with
the Newton solver.

*References:* P. Jäckel, "Let's Be Rational," *Wilmott* 2015(75)
40-53.

.. minigallery:: ../../examples/options/implied_vol/plot_05_jaeckel_lets_be_rational.py

2020 -- Negative Oil Futures: Bachelier Returns to the Exchange
---------------------------------------------------------------

On 20 April 2020 the expiring May contract for West Texas Intermediate
crude oil settled at -37.63 dollars a barrel. With storage at Cushing,
Oklahoma, nearly full, holders paid others to take delivery. Black's
model, which assumes a lognormal futures price, has no value for an
option on a negative price. CME Clearing had announced that from
22 April it would value its crude oil options with Bachelier's normal
model. So 120 years after the thesis, the first option-pricing model
was back in production, quoted in normal volatility, which is
measured in dollars rather than percent.

.. math::

   c = e^{-rT}\left[(F - K) N(d) + \sigma_N\sqrt{T}\, n(d)\right], \qquad
   d = \frac{F - K}{\sigma_N\sqrt{T}}, \quad F, K \in \mathbb{R}

*Implementation:* :func:`financialkit.options.systems.implied_vol.bachelier_implied_volatility`
inverts :func:`~financialkit.options.systems.closed_form.bachelier_price`
for any sign of the futures price and strike. The example prices options
around a futures price of -37.63, recovers the normal volatility, and
shows a flat normal volatility as a lognormal skew.

*References:* CME Group, "Switch to Bachelier Options Pricing Model --
Effective April 22, 2020," CME Clearing Advisory (April 2020);
L. Bachelier, *Annales scientifiques de l'École Normale Supérieure* 17
(1900) 21-86.

.. minigallery:: ../../examples/options/closed_form/plot_07_bachelier_negative_prices.py
