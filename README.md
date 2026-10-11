# financialkit

| | |
|:--|:-:|
| Package | [![PyPI version](https://img.shields.io/pypi/v/financialkit)](https://pypi.org/project/financialkit/) [![Python versions](https://img.shields.io/pypi/pyversions/financialkit)](https://pypi.org/project/financialkit/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23279817.svg)](https://doi.org/10.5281/zenodo.23279817) |
| Quality | [![License](https://img.shields.io/github/license/cpoli/financialkit)](https://github.com/cpoli/financialkit/blob/main/LICENSE) [![CI](https://github.com/cpoli/financialkit/actions/workflows/ci.yml/badge.svg)](https://github.com/cpoli/financialkit/actions/workflows/ci.yml) [![Coverage](https://img.shields.io/codecov/c/github/cpoli/financialkit)](https://codecov.io/gh/cpoli/financialkit) |
| Documentation | [![Docs](https://img.shields.io/badge/docs-cpoli.github.io%2Ffinancialkit-blue)](https://cpoli.github.io/financialkit/) |
| Code style | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Downloads | [![Downloads](https://static.pepy.tech/badge/financialkit)](https://pepy.tech/project/financialkit) [![Downloads/Month](https://static.pepy.tech/badge/financialkit/month)](https://pepy.tech/project/financialkit) |
| Community | [![GitHub Stars](https://img.shields.io/github/stars/cpoli/financialkit?style=social)](https://github.com/cpoli/financialkit) [![GitHub Forks](https://img.shields.io/github/forks/cpoli/financialkit?style=social)](https://github.com/cpoli/financialkit) [![Contributors](https://img.shields.io/github/contributors/cpoli/financialkit)](https://github.com/cpoli/financialkit/graphs/contributors) [![Last Commit](https://img.shields.io/github/last-commit/cpoli/financialkit)](https://github.com/cpoli/financialkit/commits/main) |
| Try it online | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cpoli/financialkit/blob/main/notebooks/quickstart.ipynb) [![JupyterLite](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://cpoli.github.io/financialkit/lite/lab/) |

**See how the models of quantitative finance actually work.**
financialkit is a Python toolkit for learning and teaching quantitative
finance: the time value of money, bonds and yield curves, option pricing,
interest-rate and credit models, Monte Carlo and PDE methods, stochastic
volatility, financial time series, risk, portfolios, asset pricing and
market microstructure. No black boxes: every result is an inspectable
dataclass, every domain ships plotting helpers, and each domain's docs walk
through the field's breakthroughs in historical order, each one linked to
runnable code that reproduces it.

> **Disclaimer.** financialkit is educational and research software. It is
> not investment advice, and it is not a production pricing or risk system.
> Its examples use synthetic data, and nothing in it should be relied on for
> trading, valuation or regulatory decisions.

![A binomial tree converging to Black-Scholes, Heston smiles and skews, and the efficient frontier, all drawn with financialkit](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_hero.png)

- **For students:** watch a binomial tree converge to Black-Scholes, see
  where the volatility smile comes from, and bootstrap a yield curve, all
  in a few lines each.
- **For instructors:** 14 domains, one consistent API, and over 200
  gallery examples, each downloadable as a notebook or runnable in the
  browser, ready to hand out as course material.
- **Built on numpy/scipy:** production-grade library routines under the
  hood, with algorithms hand-written only where the steps themselves are
  what you're learning (see [Design](#design)).

financialkit is part of a family of packages, with
[mathematicskit](https://github.com/cpoli/mathematicskit),
[physicskit](https://github.com/cpoli/physicskit) and
[chemistrykit](https://github.com/cpoli/chemistrykit), that share the same
architecture, API conventions and history-driven documentation.

## Install

```bash
pip install "financialkit[fast]"   # with numba JIT acceleration (recommended)
pip install financialkit           # pure numpy/scipy/matplotlib, e.g. for Pyodide/JupyterLite
```

Then `import financialkit as fk`. The `fast` extra adds
[numba](https://numba.pydata.org/), which compiles the inner loops of
`integrators`, `options`, `interest_rate_models`, `monte_carlo`,
`pde_methods` and `stochastic_volatility`. Without it those kernels run as
plain Python: same results, slower.

For development: `pip install -e ".[dev]"` (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Quick start

One European call, four pricers:

```python
import financialkit as fk

S, K, T, r, sigma = 100.0, 95.0, 1.0, 0.05, 0.2
print(fk.options.black_scholes_price(S, K, T, r, sigma))  # 13.3465 closed form
print(fk.options.CRRTree(1000).price(S, K, T, r, sigma).price)  # 13.3454 binomial tree
print(fk.pde_methods.CrankNicolson(400, 200).price(S, K, T, r, sigma).price)  # 13.3441 finite differences
mc = fk.monte_carlo.european_option_mc(S, K, T, r, sigma, n_paths=200_000, rng=0)
print(mc.estimate, mc.std_error)  # 13.361 +/- 0.036
```

The [quickstart notebook](https://github.com/cpoli/financialkit/blob/main/notebooks/quickstart.ipynb) tours six domains in
ten minutes. Open it in Colab using the badge above, or in the browser with
JupyterLite.

Conventions are the same everywhere: time in years, rates and volatilities
as decimals, continuous compounding unless a parameter says otherwise, and
an `rng` argument (a `numpy.random.Generator`, a seed or `None`) on every
stochastic function.

## Subpackages

Domain subpackages, grouped as in the docs, each with runnable examples
linked below.

### Foundations

- [`financialkit.time_value`](https://cpoli.github.io/financialkit/api/gallery/time_value/) -- discrete and continuous compounding and equivalent rates, present value and NPV, annuities, perpetuities and life annuities, IRR (`scipy.optimize.brentq`, with a warning when the cash flows admit several roots) and MIRR, amortization schedules and the APR, sinking funds, and day counts, year fractions and accrued interest for every `DayCount` convention.

  ![Compounding toward e, an NPV profile with two IRRs, and a mortgage amortization schedule](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_time_value.png)

- [`financialkit.fixed_income`](https://cpoli.github.io/financialkit/api/gallery/fixed_income/) -- bond prices, yields to maturity and STRIPS; Macaulay, modified, Fisher-Weil, effective and key rate durations and convexity; Redington immunization and the Fong-Vasicek bound; hand-written curve bootstrapping, McCulloch splines, and Nelson-Siegel and Svensson fits (`scipy.optimize.least_squares`); principal components of yield changes; and interest rate and currency swaps.

  ![The price-yield curve of a bond, a Nelson-Siegel zero curve, and yield-curve principal components](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_fixed_income.png)

### Derivatives

- [`financialkit.options`](https://cpoli.github.io/financialkit/api/gallery/options/) -- Bachelier, Black-Scholes-Merton, Black-76 and Garman-Kohlhagen prices that broadcast over strikes and maturities; put-call parity and no-arbitrage bounds; analytic Greeks checked against finite differences, and delta hedging; implied volatility (`brentq`, after a bounds check), the smile and the Breeden-Litzenberger density; hand-written CRR, Rendleman-Bartter, Leisen-Reimer and trinomial trees with American exercise; and the Barone-Adesi-Whaley approximation.

  ![Black-Scholes call values, the Greeks of a call, and implied volatility smiles](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_options.png)

- [`financialkit.exotic_options`](https://cpoli.github.io/financialkit/api/gallery/exotic_options/) -- Merton and Reiner-Rubinstein barriers, double and discretely monitored barriers, binaries and one-touches, geometric (Kemna-Vorst) and arithmetic (Turnbull-Wakeman, Monte Carlo with a geometric control variate) Asians, lookbacks, variance swaps, Geske compounds, forward starts and choosers, Margrabe, Stulz and quanto options, and American options by Longstaff-Schwartz.

  ![GBM paths knocked out by a double barrier, Merton's down-and-out call, and the Longstaff-Schwartz exercise boundary](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_exotic_options.png)

- [`financialkit.stochastic_volatility`](https://cpoli.github.io/financialkit/api/gallery/stochastic_volatility/) -- CEV and Dupire local volatility; Hull-White and Heston stochastic volatility; Merton, Kou and Bates jumps and variance gamma; Fourier pricing by Carr-Madan FFT (`scipy.fft`), Lewis and COS; SABR, SVI and Lee's moment formula; and Heston simulation by full-truncation Euler and Andersen's QE scheme.

  ![Heston volatility paths, Heston smiles by vol of vol, and Dupire local volatility](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_stochastic_volatility.png)

### Numerical Methods

- [`financialkit.monte_carlo`](https://cpoli.github.io/financialkit/api/gallery/monte_carlo/) -- crude Monte Carlo with standard errors and confidence intervals; stratification, importance sampling, antithetic, control and moment-matched variates, and Latin hypercubes; Halton and Sobol quasi-Monte Carlo (`scipy.stats.qmc`), randomized for error estimates, with discrepancy measures; pathwise and likelihood-ratio Greeks; and Giles's multilevel Monte Carlo.

  ![Sobol points, quasi-Monte Carlo error convergence, and Monte Carlo option prices with error bars](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_monte_carlo.png)

- [`financialkit.pde_methods`](https://cpoli.github.io/financialkit/api/gallery/pde_methods/) -- the Black-Scholes PDE by explicit, implicit and Crank-Nicolson schemes with Rannacher smoothing (`scipy.sparse` and `scipy.linalg.solve_banded`, plus a hand-written Thomas solver); CFL and von Neumann stability; Richardson extrapolation and Tavella-Randall grids; the heat-equation transform and the trinomial equivalence; American options by PSOR, Brennan-Schwartz and the penalty method; and two-factor Peaceman-Rachford ADI.

  ![A Crank-Nicolson value surface, von Neumann amplification factors, and the PSOR exercise boundary](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_pde_methods.png)

### Rates & Credit

- [`financialkit.interest_rate_models`](https://cpoli.github.io/financialkit/api/gallery/interest_rate_models/) -- Merton, Vasicek and CIR short rates (with the Feller condition) and closed-form bonds checked against Monte Carlo through `integrators`; Ho-Lee, Hull-White, G2++ and Black-Derman-Toy fitted to a `fixed_income` curve; Jamshidian bond options and swaptions; caps, floors and swaptions through the `options` Black-76; futures convexity adjustments; and the HJM drift condition.

  ![Vasicek yield curves, mean-reverting short-rate paths, and a Black-Derman-Toy tree](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_interest_rate_models.png)

- [`financialkit.credit`](https://cpoli.github.io/financialkit/api/gallery/credit/) -- Altman's Z-score; Merton, Black-Cox and Leland structural models and the KMV distance to default; hazard rates and survival curves; CDS pricing, upfronts and ISDA-style hazard bootstrapping; CreditMetrics migrations and generator matrices (`scipy.linalg.expm`); Vasicek's large pool and the Basel IRB formula; CreditRisk+ by Panjer recursion; Li's Gaussian copula; and CDO tranches.

  ![A survival curve bootstrapped from CDS spreads, a CreditRisk+ loss distribution, and a rating transition matrix](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_credit.png)

### Portfolios & Asset Pricing

- [`financialkit.portfolio`](https://cpoli.github.io/financialkit/api/gallery/portfolio/) -- Markowitz frontiers and the minimum-variance and tangency portfolios (closed forms, plus `scipy.optimize.minimize` with constraints); Tobin's two-fund separation; Kelly and Merton's portfolio problem; CPPI; single-index and Ledoit-Wolf covariances; Black-Litterman and resampled frontiers; CVaR optimization (Rockafellar-Uryasev via `scipy.optimize.linprog`); maximum diversification, risk parity and hierarchical risk parity; and the Sharpe, Sortino and information ratios.

  ![Risk-parity weights, the most diversified portfolio, and CPPI wealth paths](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_portfolio.png)

- [`financialkit.asset_pricing`](https://cpoli.github.io/financialkit/api/gallery/asset_pricing/) -- the dividend discount and Gordon growth models; Modigliani-Miller, WACC and Shiller's variance bound; the CAPM, the security market line, Jensen's alpha, the zero-beta CAPM and the GRS test; Fama-MacBeth and time-series factor regressions (`numpy.linalg.lstsq`) on synthetic Fama-French and Carhart factors; the APT; Lucas's tree, the consumption CAPM and the Mehra-Prescott equity premium puzzle; and state prices and the Hansen-Jagannathan bound.

  ![The security market line, the Hansen-Jagannathan bound, and Shiller's excess volatility](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_asset_pricing.png)

### Risk & Markets

- [`financialkit.risk`](https://cpoli.github.io/financialkit/api/gallery/risk/) -- historical, parametric, Cornish-Fisher, Monte Carlo, hybrid and filtered historical VaR; expected shortfall, coherence checks and Euler allocation; extreme value theory by block maxima and peaks over threshold (`scipy.stats.genextreme`/`genpareto`); Kupiec, Christoffersen and Basel traffic-light backtests; Gaussian and t copulas and tail dependence; safety first; and drawdowns.

  ![Monte Carlo VaR of a straddle book, a VaR backtest, and a t copula](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_risk.png)

- [`financialkit.time_series`](https://cpoli.github.io/financialkit/api/gallery/time_series/) -- returns and their stylized facts, fat tails and the Hill estimator; runs, Ljung-Box and Lo-MacKinlay variance-ratio tests; Yule-Walker and ARMA by conditional maximum likelihood with AIC order selection; the Dickey-Fuller test with MacKinnon critical values and Engle-Granger cointegration; the Hurst exponent; ARCH, GARCH, EGARCH and GJR by maximum likelihood (`scipy.optimize.minimize`) and RiskMetrics EWMA; and range-based and realized volatility.

  ![Power-law tails of returns, GARCH conditional volatility, and the autocorrelation of squared returns](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_time_series.png)

- [`financialkit.market_microstructure`](https://cpoli.github.io/financialkit/api/gallery/market_microstructure/) -- quoted, effective and realized spreads; the Roll and Corwin-Schultz estimators; Lee-Ready and bulk-volume trade classification; Garman, Ho-Stoll and Avellaneda-Stoikov dealers; Kyle and Glosten-Milgrom; Hasbrouck's VAR; PIN and VPIN; square-root impact; Bertsimas-Lo and Almgren-Chriss optimal execution; and Amihud illiquidity.

  ![Almgren-Chriss liquidation trajectories, Glosten-Milgrom quotes, and Avellaneda-Stoikov market making](https://raw.githubusercontent.com/cpoli/financialkit/main/docs/source/_static/images/readme_market_microstructure.png)

Shared infrastructure, used across the subpackages above rather than
standalone toolkits:

- `financialkit.integrators` -- the shared SDE schemes: Euler-Maruyama and
  Milstein, exact GBM and Ornstein-Uhlenbeck sampling, the Brownian bridge,
  correlated (Cholesky), antithetic and Sobol (Brownian-bridge ordered)
  increments, and strong and weak convergence tests. Every domain that
  simulates paths calls it.
- `financialkit.constants` -- trading and calendar days per year, the
  `DayCount` enum and default numerical tolerances.

## Design

financialkit calls `numpy`/`scipy` directly for anything they already
implement (distributions, root finding, optimization, linear programming,
quadrature, FFTs, quasi-random sequences, sparse solvers and matrix
exponentials), and hand-writes an algorithm only where no `numpy`/`scipy`
equivalent exists (binomial and trinomial trees, Longstaff-Schwartz, curve
bootstrapping, PSOR, the ARMA and GARCH likelihoods, the ADF test with
MacKinnon critical values, Ledoit-Wolf shrinkage) or where the iteration
itself is the lesson. There is no dependency on pandas, statsmodels,
scikit-learn, arch or QuantLib, and nothing touches the network: every
example runs on seeded synthetic data. Path simulation lives in one place,
`financialkit.integrators`, and each shared model has one implementation
(Black-76 lives in `options`, year fractions in `time_value`).

The public API follows the [stability and deprecation policy](CONTRIBUTING.md#api-stability-and-deprecation-policy).

## Test

Tests live alongside each subpackage, at `financialkit/<name>/tests/`.

```bash
pytest                                              # everything
pytest financialkit/options/tests                   # a single subpackage

# docstring examples, across every subpackage:
MPLBACKEND=Agg pytest --doctest-modules financialkit --ignore-glob="*/tests/*"
```

Monte Carlo tests are seeded and assert agreement within a few standard
errors; deterministic tests check closed forms, parity relations and
published values from Hull, Haug and the original papers. Both commands,
plus `ruff check`/`ruff format --check` and `mypy`, run in CI on every PR
(`.github/workflows/ci.yml`) across Python 3.10-3.15 on Linux and macOS.
See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

### Coverage

```bash
MPLBACKEND=Agg pytest -q --cov=financialkit --cov-report=
NUMBA_DISABLE_JIT=1 MPLBACKEND=Agg pytest -q --cov=financialkit --cov-append --cov-report=term
```

Coverage cannot trace `@njit`-compiled code, so the kernel bodies are only
covered by the second run, where they execute as plain Python. CI does the
same: the numba and no-numba jobs both upload to Codecov, which merges them.
Combined: 868 tests, 99% line coverage overall (98% excluding the test
files themselves). Per-subpackage coverage, excluding tests:

| Subpackage | Coverage | | Subpackage | Coverage |
|:--|--:|---|:--|--:|
| `time_value` | 99% | | `time_series` | 99% |
| `fixed_income` | 96% | | `risk` | 98% |
| `options` | 99% | | `portfolio` | 98% |
| `interest_rate_models` | 97% | | `asset_pricing` | 99% |
| `monte_carlo` | 100% | | `credit` | 98% |
| `exotic_options` | 99% | | `market_microstructure` | 99% |
| `pde_methods` | 99% | | `integrators` | 100% |
| `stochastic_volatility` | 99% | | `constants` | 100% |

`visualizers/` modules are smoke-tested only (correct return type and
shape) rather than covered line by line, per the testing convention in
[CLAUDE.md](CLAUDE.md).

## Docs

Built docs are hosted at <https://cpoli.github.io/financialkit/>, served from
the `gh-pages` branch. To build locally:

```bash
pip install -e ".[docs]"
cd docs && make html
```

Five [tutorials](https://cpoli.github.io/financialkit/tutorials/one_option_five_pricers.html)
follow one problem across several domains: *One option, five pricers*,
*The yield curve, end to end*, *Where the smile comes from*, *From returns
to risk* and *Credit risk, from one firm to a portfolio*.

See `docs/source/history/` for a chronology of each domain's breakthroughs,
from Fibonacci's present values to VPIN, each linked to the code that
reproduces it.

## Citation

If you use financialkit in your research, please cite it; see
[CITATION.cff](CITATION.cff). Each release is archived on Zenodo with its own
DOI; version 0.1.0 is
[10.5281/zenodo.23279817](https://doi.org/10.5281/zenodo.23279817).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please note that this project
follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

MIT -- see [LICENSE](LICENSE).
