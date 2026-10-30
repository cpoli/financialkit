Breakthroughs in Stochastic Volatility
======================================


.. include:: /_generated/nav/stochastic_volatility.rst

.. epigraph::

   "The implied volatility surface is the wrong number to put in the
   wrong formula to get the right price."
   -- attributed to Riccardo Rebonato

Black and Scholes assumed one constant volatility, and option markets
have disagreed ever since. Implied volatilities vary with the strike
(the smile, and since 1987 the skew) and with the maturity, and the
returns underneath them have fat tails, jumps and volatility that comes
in clusters. Three families of models grew out of this: local
volatility, which makes volatility a function of price and time;
stochastic volatility, which gives it its own random dynamics; and
jumps, which let prices move discontinuously. Their practical success
depended on two further ideas: pricing through the characteristic
function by Fourier inversion, and compact parametrisations of the
smile itself. This chronology traces the ideas behind
:mod:`financialkit.stochastic_volatility`.

.. contents:: Timeline
   :local:
   :depth: 1

1975 -- Cox's Constant Elasticity of Variance
---------------------------------------------

John Cox, in a Stanford working paper that circulated for twenty years
before it was published, let the instantaneous variance of returns be
a power of the price. With an elasticity :math:`\beta < 1` volatility
rises as the price falls, the leverage effect Fischer Black had just
reported: a falling share price raises a firm's debt-to-equity ratio and
with it the risk of its equity. The model produces a downward skew in
implied volatility from one extra parameter, and Mark Schroder (1989)
expressed its option prices through the noncentral chi-square
distribution.

.. math::

   dS = r S\,dt + \sigma S^{\beta}\,dW, \qquad
   \sigma_{\mathrm{loc}}(S) = \sigma S^{\beta - 1}

*Implementation:* :func:`financialkit.stochastic_volatility.systems.local_vol.cev_price`
implements Schroder's formula for :math:`\beta` below and above 1, and
:func:`~financialkit.stochastic_volatility.systems.local_vol.cev_local_vol`
gives the local volatility. The example checks the formula against an
Euler simulation and plots the skew for several elasticities.

*References:* J. C. Cox, "Notes on Option Pricing I: Constant Elasticity
of Variance Diffusions," Stanford University working paper (1975),
reprinted in *Journal of Portfolio Management* 23 (1996) 15-17;
M. Schroder, "Computing the Constant Elasticity of Variance Option
Pricing Formula," *Journal of Finance* 44 (1989) 211-219.

.. minigallery:: ../../examples/stochastic_volatility/local_vol/plot_01_cox_cev.py

1976 -- Merton's Jump Diffusion
-------------------------------

Robert Merton added a Poisson stream of jumps with lognormal sizes to
geometric Brownian motion. A jump cannot be hedged by trading the stock
continuously, so Black and Scholes's replication argument breaks down;
Merton assumed that jump risk is diversifiable across the market and
therefore unpriced. Conditioning on the number of jumps then turns the
option price into a Poisson-weighted sum of Black-Scholes prices. Jumps
give fat-tailed returns and steep short-dated smiles that flatten with
maturity.

.. math::

   C = \sum_{n \ge 0} \frac{e^{-\lambda' T}(\lambda' T)^n}{n!}\,
   C_{BS}\!\left(\sigma_n^2 = \sigma^2 + \frac{n\delta^2}{T},\;
   r_n = r - \lambda\bar k + \frac{n \ln(1 + \bar k)}{T}\right)

*Implementation:* :class:`financialkit.stochastic_volatility.systems.jumps.MertonJumpModel`
gives the characteristic function and samples terminal returns, and
:func:`~financialkit.stochastic_volatility.systems.jumps.merton_jump_price`
sums the series. The example compares the series with simulation, shows
the fat left tail and plots smiles at three maturities.

*References:* R. C. Merton, "Option Pricing When Underlying Stock
Returns Are Discontinuous," *Journal of Financial Economics* 3 (1976)
125-144.

.. minigallery:: ../../examples/stochastic_volatility/jumps/plot_01_merton_jump_diffusion.py

1987 -- Hull and White's Stochastic Volatility
----------------------------------------------

John Hull and Alan White gave the variance its own diffusion. When the
variance is independent of the stock, the stock is lognormal
conditional on the variance path, with the path's average variance; the
option price is therefore the Black-Scholes price averaged over the
distribution of that average, the *mixing* formula. Hull and White
expanded it in a series for lognormal variance. The model explained why
deep in- and out-of-the-money options trade above their Black-Scholes
values while at-the-money options trade slightly below.

.. math::

   f = \int C_{BS}(\bar V)\, h(\bar V \mid V_0)\, d\bar V, \qquad
   \bar V = \frac1T \int_0^T V_t\, dt

*Implementation:* :func:`financialkit.stochastic_volatility.systems.heston.hull_white_price`
sums the series to third order, and
:func:`~financialkit.stochastic_volatility.systems.heston.hull_white_mixing_mc`
computes the mixing integral by simulating variance paths with
:func:`financialkit.integrators.gbm_paths`. The example shows where the
series holds and the symmetric smile.

*References:* J. Hull and A. White, "The Pricing of Options on Assets
with Stochastic Volatilities," *Journal of Finance* 42 (1987) 281-300.

.. minigallery:: ../../examples/stochastic_volatility/heston/plot_01_hull_white_mixing.py

1990 -- Madan and Seneta's Variance Gamma
-----------------------------------------

Dilip Madan and Eugene Seneta ran a Brownian motion on a random clock:
business time advances by independent gamma-distributed increments, so
markets are calm on some days and frantic on others. The resulting
variance gamma process is a pure-jump Lévy process with infinitely many
small jumps and no diffusion at all. Its returns are symmetric and
leptokurtic, with semi-heavy tails, and its density is a closed form in
the modified Bessel function. It was one of the first Lévy models fitted
to stock returns.

.. math::

   X_t = \sigma W_{G_t}, \qquad G_t \sim \Gamma\!\left(\frac{t}{\nu}, \nu\right), \qquad
   E\!\left[e^{iuX_t}\right] = \left(1 + \tfrac12\sigma^2\nu u^2\right)^{-t/\nu}

*Implementation:* :class:`financialkit.stochastic_volatility.systems.jumps.VarianceGammaModel`
with ``theta = 0`` gives the characteristic function, the Bessel-function
density and sampling by subordination. The example plots the densities
for several clock variances, checks the kurtosis of simulated draws and
shows the symmetric smile.

*References:* D. B. Madan and E. Seneta, "The Variance Gamma (V.G.)
Model for Share Market Returns," *Journal of Business* 63 (1990)
511-524.

.. minigallery:: ../../examples/stochastic_volatility/jumps/plot_02_madan_seneta_variance_gamma.py

1993 -- Heston's Square-Root Stochastic Volatility
--------------------------------------------------

Steven Heston chose a variance process that mean-reverts and stays
positive, the square-root diffusion of Cox, Ingersoll and Ross,
correlated with the stock. Because the model is affine, its
characteristic function solves Riccati equations in closed form. Heston
wrote the call as the stock times one exercise probability minus the
discounted strike times another, each recovered from the characteristic
function by one Fourier integral. Negative correlation produces the
equity skew and the volatility of variance produces its curvature,
which made Heston's the reference stochastic volatility model.

.. math::

   dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_2, \qquad
   C = S P_1 - K e^{-rT} P_2, \qquad
   P_j = \frac12 + \frac1\pi \int_0^\infty \operatorname{Re}\!\left[\frac{e^{-iu\ln K} f_j(u)}{iu}\right] du

*Implementation:* :class:`financialkit.stochastic_volatility.systems.heston.HestonModel`
uses the "little trap" form of the characteristic function, and
:func:`~financialkit.stochastic_volatility.systems.heston.heston_price`
implements the two-probability formula, which reproduces Andersen's
reference prices, and
:func:`~financialkit.stochastic_volatility.systems.calibration.fourier_calibrate`
fits it, or any other characteristic-function model, to a smile by
vega-weighted least squares. The first example simulates paths and shows
how the correlation and the volatility of variance shape the smile; the
second calibrates the model to a synthetic surface, exactly and with
noisy quotes.

*References:* S. L. Heston, "A Closed-Form Solution for Options with
Stochastic Volatility with Applications to Bond and Currency Options,"
*Review of Financial Studies* 6 (1993) 327-343; H. Albrecher, P. Mayer,
W. Schoutens and J. Tistaert, "The Little Heston Trap," *Wilmott
Magazine* (January 2007) 83-92.

.. minigallery:: ../../examples/stochastic_volatility/heston/plot_02_heston.py ../../examples/stochastic_volatility/heston/plot_03_heston_calibration.py

1994 -- Dupire's Local Volatility
---------------------------------

Bruno Dupire, and independently Emanuel Derman and Iraj Kani with
implied trees, showed that the whole surface of European option prices
determines a unique diffusion whose volatility depends only on the
price and time, and that reproduces every one of those prices. The
local volatility is read off the surface with the forward equation for
call prices. It became the standard way to price exotics consistently
with the vanillas, although its smile dynamics are wrong: it predicts
that the smile moves the opposite way to the spot.

.. math::

   \sigma^2(K, T) = \frac{\partial_T C + (r - q) K\,\partial_K C + q C}{\tfrac12 K^2\,\partial_{KK} C}

*Implementation:* :func:`financialkit.stochastic_volatility.systems.local_vol.dupire_local_vol`
applies the formula to a price grid by finite differences, and
:func:`~financialkit.stochastic_volatility.systems.local_vol.local_vol_paths`
simulates the result through :func:`financialkit.integrators.euler_maruyama`.
The example recovers the CEV local volatility from CEV prices and
reprices the options by simulation.

*References:* B. Dupire, "Pricing with a Smile," *Risk* 7(1) (1994)
18-20; E. Derman and I. Kani, "Riding on a Smile," *Risk* 7(2) (1994)
32-39.

.. minigallery:: ../../examples/stochastic_volatility/local_vol/plot_02_dupire.py

1996 -- Bates: Jumps with Stochastic Volatility
-----------------------------------------------

David Bates, studying Deutsche mark currency options around the 1987
crash, found that stochastic volatility could fit long-dated smiles but
not short-dated ones, which need jumps. Adding Merton's lognormal jumps
to Heston's model gave both: the jumps steepen the short end and the
stochastic variance the long end. Because the two components are
independent, the characteristic function is simply the product of the
two, and Fourier pricing carries over unchanged.

.. math::

   \phi_{\mathrm{Bates}}(u) = \phi_{\mathrm{Heston}}(u)\,
   \exp\!\left(\lambda T\left[e^{iu\mu_J - \delta^2 u^2/2} - 1 - iu\bar k\right]\right)

*Implementation:* :class:`financialkit.stochastic_volatility.systems.jumps.BatesModel`
multiplies the Heston and Merton characteristic functions. The example
compares Heston and Bates smiles at three maturities.

*References:* D. S. Bates, "Jumps and Stochastic Volatility: Exchange
Rate Processes Implicit in Deutsche Mark Options," *Review of Financial
Studies* 9 (1996) 69-107.

.. minigallery:: ../../examples/stochastic_volatility/jumps/plot_03_bates.py

1998 -- Madan, Carr and Chang's Skewed Variance Gamma
-----------------------------------------------------

Madan, Carr and Chang gave the variance gamma process a drift
:math:`\theta` in business time. A negative drift puts more mass in the
left tail, so the symmetric Madan-Seneta smile becomes a skew, and the
process has three parameters that control volatility, skewness and
kurtosis almost separately. They priced European options in closed
form and showed that the skewed model fits S&P 500 options far better
than Black-Scholes. Fang and Oosterlee later used one of their
parameter sets as a benchmark for Fourier methods.

.. math::

   X_t = \theta G_t + \sigma W_{G_t}, \qquad
   E\!\left[e^{iuX_t}\right] = \left(1 - iu\theta\nu + \tfrac12\sigma^2\nu u^2\right)^{-t/\nu}

*Implementation:* :class:`financialkit.stochastic_volatility.systems.jumps.VarianceGammaModel`
takes the drift ``theta``, includes the martingale correction and gives
the critical moments of the price. The example reproduces Fang and
Oosterlee's reference prices and shows how :math:`\theta` skews the
density and the smile.

*References:* D. B. Madan, P. P. Carr and E. C. Chang, "The Variance
Gamma Process and Option Pricing," *European Finance Review* 2 (1998)
79-105.

.. minigallery:: ../../examples/stochastic_volatility/jumps/plot_04_madan_carr_chang.py

1999 -- Carr and Madan's FFT
----------------------------

Peter Carr and Dilip Madan observed that the call price, as a function
of the log strike, is not square integrable, but that damping it by
:math:`e^{\alpha k}` makes it so, and that the Fourier transform of the
damped price is a simple expression in the characteristic function of
the log price. A single fast Fourier transform then gives call prices
at thousands of strikes at once, for any model with a known
characteristic function. It turned Heston, variance gamma and the other
Lévy models into practical tools for calibration.

.. math::

   C(k) = \frac{e^{-\alpha k}}{\pi} \int_0^\infty e^{-ivk}
   \frac{e^{-rT}\phi_{\ln S_T}(v - (\alpha + 1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha + 1)v}\, dv

*Implementation:* :func:`financialkit.stochastic_volatility.systems.fourier.carr_madan_fft`
evaluates the integral with :func:`scipy.fft.fft` and Simpson weights,
and :func:`~financialkit.stochastic_volatility.systems.fourier.carr_madan_price`
interpolates the grid. The example checks the grid against
Black-Scholes and scans the damping exponent.

*References:* P. Carr and D. B. Madan, "Option Valuation Using the Fast
Fourier Transform," *Journal of Computational Finance* 2(4) (1999)
61-73.

.. minigallery:: ../../examples/stochastic_volatility/fourier/plot_01_carr_madan_fft.py

2001 -- Lewis's Single-Integral Formula
---------------------------------------

Alan Lewis treated the option payoff and the characteristic function as
generalised Fourier transforms, each living on its own strip of the
complex plane. Moving the inversion contour to
:math:`\operatorname{Im} u = -\tfrac12`, between the call and the put
strips, gives a single real integral whose integrand decays like
:math:`u^{-2}` for any model. The formula needs no damping parameter and
is robust for pure-jump models, where Heston's two integrals converge
slowly.

.. math::

   C = S e^{-qT} - \frac{\sqrt{FK}\,e^{-rT}}{\pi} \int_0^\infty
   \operatorname{Re}\!\left[e^{iu\ln(F/K)}\phi\!\left(u - \tfrac i2\right)\right] \frac{du}{u^2 + \tfrac14}

*Implementation:* :func:`financialkit.stochastic_volatility.systems.fourier.lewis_price`
evaluates the integral with :func:`scipy.integrate.quad`. The example
compares its integrand and prices with Heston's formula, for Heston and
for variance gamma.

*References:* A. L. Lewis, "A Simple Option Formula for General
Jump-Diffusion and Other Exponential Lévy Processes," working paper
(2001); A. L. Lewis, *Option Valuation under Stochastic Volatility*
(Finance Press, 2000).

.. minigallery:: ../../examples/stochastic_volatility/fourier/plot_02_lewis_formula.py

2002 -- Hagan's SABR Model
--------------------------

Patrick Hagan, Deep Kumar, Andrew Lesniewski and Diana Woodward showed
that local volatility predicts the wrong dynamics for the smile and
proposed the stochastic :math:`\alpha\beta\rho` model instead, together
with a singular-perturbation formula for its Black implied volatility.
The formula is explicit, fast and accurate for the strikes that matter,
and each parameter has one job: :math:`\beta` the backbone,
:math:`\rho` the skew, :math:`\nu` the curvature. SABR became the market
standard for interest-rate smiles.

.. math::

   dF = \alpha F^{\beta}\,dW_1, \qquad d\alpha = \nu\alpha\,dW_2, \qquad
   \sigma_B(F) = \frac{\alpha}{F^{1-\beta}}\left[1 + \left(\frac{(1-\beta)^2\alpha^2}{24F^{2-2\beta}}
   + \frac{\rho\beta\nu\alpha}{4F^{1-\beta}} + \frac{2 - 3\rho^2}{24}\nu^2\right)T\right]

*Implementation:* :func:`financialkit.stochastic_volatility.systems.smile.sabr_implied_vol`
implements Hagan's formula, and
:func:`~financialkit.stochastic_volatility.systems.smile.sabr_calibrate`
fits it with :func:`scipy.optimize.least_squares`. The example calibrates
a noisy smile and varies each parameter.

*References:* P. S. Hagan, D. Kumar, A. S. Lesniewski and D. E. Woodward,
"Managing Smile Risk," *Wilmott Magazine* (September 2002) 84-108.

.. minigallery:: ../../examples/stochastic_volatility/smile/plot_01_sabr.py

2002 -- Kou's Double-Exponential Jumps
--------------------------------------

Steven Kou replaced Merton's normal jump sizes with an asymmetric
double exponential: upward jumps at one rate, downward at another. The
model reproduces both the asymmetric fat tails of returns and the
smile, and because the exponential distribution is memoryless, the
overshoot of a barrier is again exponential, so barrier, lookback and
American options have analytic solutions. The exponential tails also
cap the moments of the price, which ties the model to Lee's wing
formula two years later.

.. math::

   f_Y(y) = p\,\eta_1 e^{-\eta_1 y}\,\mathbf 1_{y \ge 0} + (1 - p)\,\eta_2 e^{\eta_2 y}\,\mathbf 1_{y < 0}

*Implementation:* :class:`financialkit.stochastic_volatility.systems.jumps.KouModel`
gives the characteristic function, the critical moments and sampling by
gamma-distributed jump totals. The example reproduces Kou's published
price and compares exponential with Gaussian jump tails.

*References:* S. G. Kou, "A Jump-Diffusion Model for Option Pricing,"
*Management Science* 48 (2002) 1086-1101.

.. minigallery:: ../../examples/stochastic_volatility/jumps/plot_05_kou.py

2004 -- Gatheral's SVI
----------------------

Jim Gatheral, presenting Merrill Lynch's practice at the Global
Derivatives conference in Madrid, introduced the stochastic volatility
inspired parametrisation of total implied variance. It is a hyperbola
in log-moneyness, whose asymptotes are the straight wings that Lee's
theorem requires, and it is the large-maturity limit of Heston's
implied variance. Five parameters fit equity smiles closely, and later
work with Antoine Jacquier gave conditions under which an SVI surface is
free of static arbitrage.

.. math::

   w(k) = a + b\left[\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right]

*Implementation:* :func:`financialkit.stochastic_volatility.systems.smile.svi_total_variance`
and :func:`~financialkit.stochastic_volatility.systems.smile.svi_fit`
evaluate and fit raw SVI, and
:func:`~financialkit.stochastic_volatility.systems.smile.durrleman_g`
checks for butterfly arbitrage. The example fits SVI to Heston smiles at
three maturities.

*References:* J. Gatheral, "A Parsimonious Arbitrage-Free Implied
Volatility Parameterization with Application to the Valuation of
Volatility Derivatives," Global Derivatives & Risk Management, Madrid
(2004); J. Gatheral and A. Jacquier, "Arbitrage-Free SVI Volatility
Surfaces," *Quantitative Finance* 14 (2014) 59-71.

.. minigallery:: ../../examples/stochastic_volatility/smile/plot_02_svi.py

2004 -- Lee's Moment Formula
----------------------------

Roger Lee proved, without assuming any model, that total implied
variance grows at most linearly in the log strike, with a slope no
greater than 2, and that the exact slope is determined by the number of
finite moments of the price. Heavy right tails, with few finite moments,
give steep right wings. The result constrains every smile
parametrisation and extrapolation, and it reads a model's tail
behaviour straight off its smile.

.. math::

   \limsup_{k \to \infty} \frac{\sigma_{BS}^2(k)\,T}{k} = 2 - 4\left(\sqrt{\tilde p^2 + \tilde p} - \tilde p\right), \qquad
   \tilde p = \sup\{p : E[S_T^{1+p}] < \infty\}

*Implementation:* :func:`financialkit.stochastic_volatility.systems.smile.lee_wing_slope`
evaluates the formula, and
:meth:`KouModel.critical_moments <financialkit.stochastic_volatility.systems.jumps.KouModel.critical_moments>`
and :meth:`VarianceGammaModel.critical_moments <financialkit.stochastic_volatility.systems.jumps.VarianceGammaModel.critical_moments>`
supply the moments. The example shows Kou smiles approaching Lee's
bounds in both wings.

*References:* R. W. Lee, "The Moment Formula for Implied Volatility at
Extreme Strikes," *Mathematical Finance* 14 (2004) 469-480.

.. minigallery:: ../../examples/stochastic_volatility/smile/plot_03_lee_moment_formula.py

2008 -- Andersen's Quadratic-Exponential Scheme
-----------------------------------------------

Leif Andersen addressed the practical problem of simulating Heston when
the Feller condition fails, as it does in most calibrations, so that
the variance often reaches zero. Rather than discretising the SDE, he
matched the first two moments of the exact noncentral chi-square
transition with a scaled squared Gaussian when the variance is large
and with a point mass at zero plus an exponential tail when it is
small. Combined with a careful step for the stock, the scheme is nearly
unbiased at a few steps per year on test cases where Euler schemes are
badly biased.

.. math::

   v_{t+\Delta} = \begin{cases} a(b + Z)^2, & \psi \le \psi_c \\
   0 \text{ w.p. } p, \text{ else } \mathrm{Exp}(\beta), & \psi > \psi_c \end{cases}, \qquad
   \psi = \frac{\operatorname{Var}[v_{t+\Delta}]}{E[v_{t+\Delta}]^2}

*Implementation:* :func:`financialkit.stochastic_volatility.systems.simulation.heston_paths`
with ``scheme="qe"`` implements the scheme, and
:func:`~financialkit.stochastic_volatility.systems.simulation.heston_mc_price`
prices with it. The example measures its bias against full truncation on
Andersen's Case I and shows the atom at zero.

*References:* L. Andersen, "Simple and Efficient Simulation of the Heston
Stochastic Volatility Model," *Journal of Computational Finance* 11(3)
(2008) 1-42.

.. minigallery:: ../../examples/stochastic_volatility/simulation/plot_01_andersen_qe.py

2008 -- Fang and Oosterlee's COS Method
---------------------------------------

Fang Fang and Kees Oosterlee truncated the density of the log return to
an interval set by its cumulants and expanded it in a Fourier cosine
series. The series coefficients come straight from the characteristic
function, and the payoff's coefficients are closed form, so a price is
a short dot product. For smooth densities the error falls exponentially
with the number of terms, which made COS faster than the FFT for most
models, and it extends to Bermudan and barrier options.

.. math::

   V = e^{-rT} \sum_{k=0}^{N-1}{}' \operatorname{Re}\!\left[\phi\!\left(\frac{k\pi}{b - a}\right) e^{-ik\pi a/(b - a)}\right] V_k

*Implementation:* :func:`financialkit.stochastic_volatility.systems.fourier.cos_price`
prices puts by the series and calls by parity, with the truncation
interval from :func:`~financialkit.stochastic_volatility.systems.fourier.truncation_range`,
and :func:`~financialkit.stochastic_volatility.systems.fourier.cos_density`
recovers densities. The example shows exponential convergence for
Heston and slower convergence for short-dated variance gamma.

*References:* F. Fang and C. W. Oosterlee, "A Novel Pricing Method for
European Options Based on Fourier-Cosine Series Expansions," *SIAM
Journal on Scientific Computing* 31 (2008) 826-848.

.. minigallery:: ../../examples/stochastic_volatility/fourier/plot_03_cos_method.py

2010 -- Lord, Koekkoek and van Dijk's Full Truncation
-----------------------------------------------------

An Euler step can push the square-root variance below zero, and the
literature had proposed several repairs: absorbing at zero, reflecting,
taking absolute values, or truncating in the diffusion only. Roger Lord,
Remmert Koekkoek and Dick van Dijk put them all in one framework and
proposed *full truncation*: let the variance go negative but use its
positive part in both the drift and the diffusion. It had the smallest
bias of the Euler schemes in their tests and remains the default simple
scheme.

.. math::

   v_{t+\Delta} = v_t + \kappa(\theta - v_t^+)\,\Delta + \xi\sqrt{v_t^+}\,\Delta W

*Implementation:* :func:`financialkit.stochastic_volatility.systems.simulation.heston_paths`
with ``scheme="full_truncation"``, ``"partial_truncation"`` or
``"higham_mao"`` runs the Euler schemes through
:func:`financialkit.integrators.euler_maruyama` with correlated increments.
The example compares their biases on Andersen's Case I.

*References:* R. Lord, R. Koekkoek and D. van Dijk, "A Comparison of
Biased Simulation Schemes for Stochastic Volatility Models,"
*Quantitative Finance* 10 (2010) 177-194.

.. minigallery:: ../../examples/stochastic_volatility/simulation/plot_02_full_truncation.py

2014 -- Gatheral and Jacquier's Arbitrage-Free SSVI Surfaces
------------------------------------------------------------

Fitting SVI one expiry at a time leaves the slices free to cross, which
is calendar-spread arbitrage, and lets a single slice imply a negative
density, which is butterfly arbitrage. Jim Gatheral and Antoine Jacquier
parametrised the whole surface at once. The at-the-money total variance
:math:`\theta_T` carries the term structure, and one skew :math:`\rho` and
one curvature function :math:`\varphi(\theta)` shape every slice. They
proved that the surface is free of calendar arbitrage when
:math:`\theta\varphi(\theta)` rises, but not too fast, and free of
butterfly arbitrage when :math:`\theta\varphi(1 + |\rho|) < 4` and
:math:`\theta\varphi^2(1 + |\rho|) \le 4`. Simple parameter bounds, such
as :math:`\eta(1 + |\rho|) \le 2` for a power-law :math:`\varphi`, then
guarantee no static arbitrage, and each SSVI slice is still a raw SVI
slice.

.. math::

   w(k, \theta_T) = \frac{\theta_T}{2}\left[1 + \rho\varphi(\theta_T)k
   + \sqrt{(\varphi(\theta_T)k + \rho)^2 + 1 - \rho^2}\right], \qquad
   \varphi(\theta) = \frac{\eta}{\theta^{\gamma}(1 + \theta)^{1-\gamma}}

*Implementation:* :class:`financialkit.stochastic_volatility.systems.surface.SSVISurface`
evaluates the surface with power-law or Heston-like curvature, checks
Gatheral and Jacquier's conditions and converts its slices to raw SVI,
and :func:`~financialkit.stochastic_volatility.systems.surface.ssvi_surface`
fits it with the no-arbitrage bounds built into the parameters. The
example fits Heston smiles at four expiries, confirms the result with
Durrleman's function on every slice, and breaks the butterfly condition
with a steeper curvature.

*References:* J. Gatheral and A. Jacquier, "Arbitrage-Free SVI
Volatility Surfaces," *Quantitative Finance* 14 (2014) 59-71.

.. minigallery:: ../../examples/stochastic_volatility/smile/plot_04_ssvi.py

2016 -- Bayer, Friz and Gatheral's Rough Bergomi
------------------------------------------------

In equity markets the at-the-money skew grows roughly like
:math:`\tau^{-0.4}` as the maturity :math:`\tau` shrinks, while Markovian
stochastic volatility models such as Heston give a skew that levels
off. Christian Bayer, Peter Friz and Jim Gatheral replaced the
variance's Brownian driver in Bergomi's forward-variance model with a
Riemann-Liouville fractional process whose kernel
:math:`(t - s)^{H - 1/2}` is singular for :math:`H < 1/2`. The skew then
scales like :math:`\tau^{H - 1/2}`. With :math:`H \approx 0.07` and only
three parameters beyond the forward variance curve, the model fitted the
whole S&P 500 surface. They priced by simulating the Gaussian pair of
the Volterra process and its Brownian driver exactly, by Cholesky
factorisation of their joint covariance.

.. math::

   v_t = \xi_0(t)\exp\left(\eta\sqrt{2H}\int_0^t (t - s)^{H - 1/2}\,dW_s - \tfrac12\eta^2 t^{2H}\right), \qquad
   \frac{dS_t}{S_t} = \sqrt{v_t}\,\left(\rho\,dW_t + \sqrt{1 - \rho^2}\,dW_t^{\perp}\right)

*Implementation:* :class:`financialkit.stochastic_volatility.systems.rough.RoughBergomiModel`
holds the model,
:func:`~financialkit.stochastic_volatility.systems.rough.volterra_covariance`
gives the joint covariance through the hypergeometric function,
:func:`~financialkit.stochastic_volatility.systems.rough.rough_bergomi_paths`
with ``scheme="cholesky"`` simulates it exactly, and
:func:`~financialkit.stochastic_volatility.systems.rough.rough_bergomi_mc_price`
prices by Monte Carlo. The example plots smiles with Bayer, Friz and
Gatheral's parameters and fits the power law of the at-the-money skew
against Heston's.

*References:* C. Bayer, P. Friz and J. Gatheral, "Pricing Under Rough
Volatility," *Quantitative Finance* 16 (2016) 887-904; L. Bergomi,
"Smile Dynamics II," *Risk* 18(10) (2005) 67-73.

.. minigallery:: ../../examples/stochastic_volatility/rough/plot_01_rough_bergomi.py

2017 -- Bennedsen, Lunde and Pakkanen's Hybrid Scheme
-----------------------------------------------------

Exact sampling of a Volterra process costs :math:`O(n^2)` per path on
:math:`n` steps, and the obvious Riemann sum fails near the singularity
of the kernel: for small :math:`H` it loses much of the variance. Mikkel
Bennedsen, Asger Lunde and Mikko Pakkanen integrated the kernel exactly
over the last few steps, as Wiener integrals jointly Gaussian with the
Brownian increments, and replaced it elsewhere by its value at optimally
chosen points inside each step. The remaining sum is a discrete
convolution, computed by FFT in :math:`O(n \log n)` per path. The
scheme is nearly exact for every :math:`H` and became the standard way
to simulate rough volatility models.

.. math::

   \tilde W_{t_i} \approx \sqrt{2H}\left[\int_{t_{i-1}}^{t_i}(t_i - s)^{\alpha}\,dW_s
   + \sum_{k=2}^{i} (b_k\Delta)^{\alpha}\,\Delta W_{i-k}\right], \qquad
   b_k = \left(\frac{k^{\alpha+1} - (k-1)^{\alpha+1}}{\alpha + 1}\right)^{1/\alpha}

*Implementation:* :func:`financialkit.stochastic_volatility.systems.rough.volterra_paths`
implements the hybrid scheme with :math:`\kappa = 1` and the convolution
by :mod:`scipy.fft`, alongside exact Cholesky sampling and the Riemann
sum, and
:func:`~financialkit.stochastic_volatility.systems.rough.hybrid_scheme_variance`
gives each scheme's variance from its weights. The example compares the
variances with :math:`t^{2H}`, rough Bergomi prices across the three
schemes, and paths for three Hurst exponents.

*References:* M. Bennedsen, A. Lunde and M. S. Pakkanen, "Hybrid Scheme
for Brownian Semistationary Processes," *Finance and Stochastics* 21
(2017) 931-965; R. McCrickerd and M. S. Pakkanen, "Turbocharging Monte
Carlo Pricing for the Rough Bergomi Model," *Quantitative Finance* 18
(2018) 1877-1886.

.. minigallery:: ../../examples/stochastic_volatility/rough/plot_02_hybrid_scheme.py

2018 -- Gatheral, Jaisson and Rosenbaum: Volatility Is Rough
------------------------------------------------------------

Jim Gatheral, Thibault Jaisson and Mathieu Rosenbaum measured how
increments of log-volatility, estimated from high-frequency realized
variance, scale with the lag. For every moment :math:`q` the empirical
moments lay on a straight line in the log lag, with slopes proportional
to :math:`q`, the signature of fractional Brownian motion. The Hurst
exponent was about 0.1 across equity indices and decades, far below the
:math:`1/2` of Brownian motion. Their rough fractional stochastic
volatility model, log-volatility as a fractional Brownian motion, also
explained the apparent long memory of volatility and forecast it well.
Their paper first circulated in 2014 and gave the rough volatility
models their empirical basis.

.. math::

   E\left[|\log\sigma_{t+\Delta} - \log\sigma_t|^q\right] \propto \Delta^{qH}, \qquad
   \log\sigma_t = \nu W^H_t + \text{const}, \qquad H \approx 0.1

*Implementation:* :func:`financialkit.stochastic_volatility.systems.rough.volatility_roughness`
computes the moments and fits :math:`\zeta_q = qH`,
:func:`~financialkit.stochastic_volatility.systems.rough.fbm_paths`
samples fractional Brownian motion exactly by Davies and Harte's
circulant embedding, and
:func:`financialkit.stochastic_volatility.visualizers.plot_roughness`
draws the scaling plot. The example recovers :math:`H` from a synthetic
RFSV log-volatility and contrasts it with Heston's log-volatility.

*References:* J. Gatheral, T. Jaisson and M. Rosenbaum, "Volatility Is
Rough," *Quantitative Finance* 18 (2018) 933-949; R. B. Davies and
D. S. Harte, "Tests for Hurst Effect," *Biometrika* 74 (1987) 95-101.

.. minigallery:: ../../examples/stochastic_volatility/rough/plot_03_volatility_is_rough.py
