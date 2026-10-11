# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Five cross-domain tutorials in `docs/source/tutorials/`, each linked
  from the galleries of the domains it uses: *One option, five pricers*
  (the Black-Scholes formula, trees, Monte Carlo, Crank-Nicolson and
  Fourier pricing agree, then check each other on an American put), *The
  yield curve, end to end* (bootstrapping, Nelson-Siegel, swaps,
  Hull-White and caps), *Where the smile comes from* (implied volatility,
  Heston, the Breeden-Litzenberger density, jumps and SABR), *From returns
  to risk* (stylized facts, GARCH, VaR backtests, expected shortfall and
  extreme value theory) and *Credit risk, from one firm to a portfolio*
  (Merton, KMV, hazard rates and CDS, Vasicek, Li's copula and the Basel
  formula).
- `notebooks/quickstart.ipynb`, a ten-minute tour of six domains, with a
  Colab badge in the README and the docs. It also ships in the JupyterLite
  site, so the "Try it online" badge opens on it.

## [0.1.0] - 2026-10-10

The first release.

### Added

- Repository scaffold, structurally cloned from mathematicskit:
  `pyproject.toml` (numpy, scipy and matplotlib, with numba in the `fast`
  extra), CI, docs and release workflows, pre-commit, the Sphinx site
  with per-domain hubs, galleries, history pages and JupyterLite, and the
  root docs. The README and the docs index state that the package is
  educational and research software, not investment advice.
- `financialkit.constants`: `TRADING_DAYS_PER_YEAR`,
  `CALENDAR_DAYS_PER_YEAR`, the `DayCount` enum and the default
  tolerances `DEFAULT_ATOL`, `DEFAULT_RTOL` and `DEFAULT_MAX_ITER`.
- `financialkit._jit`: an optional numba `njit` that falls back to a
  no-op decorator, so the package runs under Pyodide.
- `financialkit.integrators`, the shared SDE module: `euler_maruyama` and
  `milstein`, exact `gbm_paths` and `ou_paths`, `brownian_increments`,
  `correlated_increments`, `antithetic_increments`, `sobol_increments`
  and `brownian_bridge`, and `strong_convergence` and
  `weak_convergence` (`ConvergenceResult`).
- 14 domain subpackages. Each has its own tests, visualizers, gallery
  and a history page of at least 15 dated breakthroughs, each linked to
  its own gallery example (239 examples in all):
  - `time_value`: compounding and equivalent rates, present value and
    NPV, annuities, perpetuities and life annuities, IRR (with
    `MultipleIRRWarning`) and MIRR, amortization and the APR, sinking
    funds, and day counts and accrued interest for every `DayCount`.
  - `fixed_income`: bond prices, yields and STRIPS; Macaulay, modified,
    Fisher-Weil, effective and key rate durations and convexity;
    Redington immunization; bootstrapped, McCulloch spline,
    Nelson-Siegel and Svensson curves; yield-curve PCA; and interest
    rate and currency swaps.
  - `options`: Bachelier, Black-Scholes-Merton, Black-76 and
    Garman-Kohlhagen; parity and no-arbitrage bounds; the Greeks and
    delta hedging; implied volatility, the smile and the
    Breeden-Litzenberger density; CRR, Rendleman-Bartter, Leisen-Reimer
    and trinomial trees; and Barone-Adesi-Whaley.
  - `interest_rate_models`: Merton, Vasicek, CIR and affine short rates;
    Ho-Lee, Hull-White, G2++ and Black-Derman-Toy fitted to the curve;
    Jamshidian bond options and swaptions; Black caps, floors and
    swaptions; and HJM.
  - `monte_carlo`: crude Monte Carlo with confidence intervals;
    stratification, importance sampling, antithetic, control and
    moment-matched variates and Latin hypercubes; Halton and Sobol QMC
    and discrepancies; pathwise and likelihood-ratio Greeks; and
    multilevel Monte Carlo.
  - `exotic_options`: single, double and discrete barriers, binaries
    and one-touches, geometric and arithmetic Asians, lookbacks,
    variance swaps, compound, forward-start and chooser options,
    Margrabe, Stulz and quanto options, and Longstaff-Schwartz.
  - `pde_methods`: explicit, implicit and Crank-Nicolson schemes with
    Rannacher smoothing, the Thomas algorithm, CFL and von Neumann
    stability, Richardson extrapolation, Tavella-Randall grids, the
    heat-equation transform, PSOR, Brennan-Schwartz, the penalty method
    and Peaceman-Rachford ADI.
  - `stochastic_volatility`: CEV and Dupire local volatility,
    Hull-White and Heston, Merton, Kou, Bates and variance gamma,
    Carr-Madan FFT, Lewis and COS pricing, SABR, SVI and Lee's moment
    formula, and full-truncation and QE simulation.
  - `time_series`: stylized facts and the Hill estimator, random-walk
    tests, Yule-Walker and ARMA, Dickey-Fuller and Engle-Granger, the
    Hurst exponent, ARCH, GARCH, EGARCH, GJR and EWMA, and range and
    realized volatility.
  - `risk`: historical, parametric, Cornish-Fisher, Monte Carlo, hybrid
    and filtered VaR, expected shortfall and coherence, Euler
    allocation, extreme value theory, VaR backtests, copulas and tail
    dependence, safety first and drawdowns.
  - `portfolio`: Markowitz, Tobin and Merton, Kelly, CPPI, single-index
    and Ledoit-Wolf covariances, Black-Litterman and resampling, CVaR
    optimization, maximum diversification, risk parity and HRP, and
    performance ratios.
  - `asset_pricing`: dividend discount and Gordon growth,
    Modigliani-Miller and Shiller's variance bound, the CAPM, Jensen's
    alpha, the zero-beta CAPM and the GRS test, Fama-MacBeth, the APT
    and factor models, Lucas's tree, the consumption CAPM and the
    equity premium puzzle, and the Hansen-Jagannathan bound.
  - `credit`: Altman's Z-score, Merton, Black-Cox and Leland, the KMV
    distance to default, hazard curves, CDS pricing and bootstrapping,
    CreditMetrics and generator matrices, Vasicek's large pool,
    CreditRisk+, Li's copula, the Basel IRB formula and CDO tranches.
  - `market_microstructure`: quoted, effective and realized spreads,
    the Roll and Corwin-Schultz estimators, Lee-Ready classification,
    Garman, Ho-Stoll and Avellaneda-Stoikov dealers, Kyle and
    Glosten-Milgrom, Hasbrouck's VAR, PIN and VPIN, square-root impact,
    Bertsimas-Lo and Almgren-Chriss execution, and Amihud illiquidity.

[Unreleased]: https://github.com/cpoli/financialkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cpoli/financialkit/releases/tag/v0.1.0
