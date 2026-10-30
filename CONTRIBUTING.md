# Contributing to financialkit

Thanks for considering a contribution. financialkit is organized as one
subpackage per finance domain (`financialkit/<name>/`), each with its own
`core/` (ABCs and shared engine machinery, plus kernels JIT-compiled
by numba, when it is installed, where needed), `systems/` (concrete models/algorithms), `utils/`
(supporting numerics), `visualizers/` (matplotlib plotting), and
`tests/` directory. New models belong in the subpackage they fit
best; a genuinely new domain gets its own subpackage following the same
layout.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before opening a PR

```bash
ruff check .                 # lint
ruff format .                # format
MPLBACKEND=Agg pytest -q     # unit tests
MPLBACKEND=Agg pytest --doctest-modules financialkit --ignore-glob="*/tests/*"   # docstring examples
```

All three run in CI (`.github/workflows/ci.yml`) on every PR, across
Python 3.10-3.15 on Linux and macOS. `mypy` also runs in CI and must
pass — see "Type checking" below.

If you touch anything under `docs/` or add/modify an example in
`examples/`, also build the docs locally before opening a PR (this
re-executes every changed `examples/*/plot_*.py` script via
sphinx-gallery, which is the only way to catch a broken example):

```bash
pip install -e ".[docs]"
cd docs && make html
```

## Code style

- Follow the existing style in the subpackage you're editing —
  financial and mathematical notation (`S`, `K`, `T`, `r`, `q`, `n`,
  single-letter variables) is intentional and allowed (see the `ruff` ignores in `pyproject.toml` for
  the specific, deliberate exceptions).
- Every public function/class gets a NumPy-style docstring with a
  `Parameters`, `Returns`, and (where it adds real value beyond what the
  signature already says) an `Examples` section with a runnable doctest.
  Doctests are checked in CI — a docstring example that doesn't actually
  run is worse than no example.
- Prefer closed-form / analytically-verifiable results in tests
  (`pytest.approx` against a known formula, a parity relation or a
  published worked example) over snapshot-testing plot output. Monte
  Carlo tests are seeded and assert agreement within a few standard
  errors, never a bare fixed tolerance.
- New algorithms should cite where the result/formula comes from (a
  textbook name/edition or a paper), either in the docstring or as a
  comment, so a reader can verify it independently.
- Call `numpy`/`scipy` directly for anything they already implement
  (decompositions, eigensolvers, linear/iterative solvers, root finders,
  quadrature, FFT, statistical distributions, optimization routines,
  etc.) rather than reimplementing it — financialkit's value-add is its own
  dataclass-result API, docstrings, visualizers, tests, and examples
  wrapped around those calls, not reinventing numerical primitives that
  are already correct and well-tested upstream. Only hand-write an
  algorithm from scratch when it has **no numpy/scipy equivalent at
  all** (e.g. binomial/trinomial trees, Longstaff-Schwartz, curve
  bootstrapping, PSOR, the ARMA and GARCH likelihoods, the ADF test with
  MacKinnon critical values, Ledoit-Wolf shrinkage), or when the
  algorithm itself is the pedagogical subject of a `systems/` module.
  When in doubt, prefer the library call. No dependency on pandas,
  statsmodels, scikit-learn, arch or QuantLib, and no network access at
  runtime, in tests or in the gallery: examples use seeded synthetic
  data. `sympy` may be
  used sparingly inside *tests* for symbolic cross-checks, never as the
  runtime engine behind a public API.

## Type checking

`mypy` is configured in `pyproject.toml` and `mypy financialkit` is
clean; CI fails on any new type error. Function bodies without
annotations are not checked (`check_untyped_defs = false`), so annotate
new public functions to get them checked.

## Tests

Tests live alongside each subpackage (`financialkit/<name>/tests/`), not in a
top-level `tests/` directory. A new system/model needs:

- At least one test that checks a closed-form / analytically-known
  result, not just "it runs without crashing."
- A visualizer needs only a smoke test (it returns the right type/shape,
  and — for animations — that `anim.save()` to a temp file succeeds).
- numba is optional (the `fast` extra). Import `njit` from
  `financialkit._jit`, never from `numba` directly, so the kernel still
  runs as plain Python without numba, and CI's no-numba job still passes.
  A test that is only practical with the JIT (minutes of pure-Python
  loops) may be marked
  `@pytest.mark.skipif(not HAS_NUMBA, reason=...)`.

## Public API conventions

These rules come from the review of every `__all__` before 1.0
(ROADMAP C10, 2026-10-27). New code follows them; existing names that
broke them were renamed, with the old spelling kept as a deprecated
alias (see the next section and `CHANGELOG.md`).

**Functions or classes.** Both forms stay, each for its own job:

- A model with parameters that several operations share is a class,
  usually a frozen dataclass: `HestonModel`, `VasicekModel`, `GARCH`,
  `LucasTree`. Its methods are the operations every instance has
  (`characteristic_function`, `simulate`, `bond_price`, `variance`).
- A pricer, estimator or test is a function. A pricer that works on a
  model takes the model as an argument (`heston_price(model, ...)`,
  `cos_price(model, ...)`), so one method serves every model with the
  right interface. `heston_price` and `HestonModel` are therefore not
  rival spellings.
- A fit returns a result dataclass. It is a `fit` classmethod when the
  model is a class (`GARCH.fit`), and a `fit_<model>` function when the
  model exists only as its result (`fit_arma`, `fit_var`, `fit_hawkes`).
- Results are dataclasses, never bare tuples.

**Parameter names.** One spelling for each meaning:

| Meaning | Name | Not |
|---|---|---|
| Simulated paths, scenarios or replications of a random draw | `n_paths` | `n_sims` |
| Time steps of a path, tree or exercise grid | `n_steps` | `n_exercise` |
| The two axes of a finite-difference grid | `n_space`, `n_time` | |
| Sample size | `n_obs` | `nobs` |
| Stopping tolerance of an iterative solver | `tol` | `rtol` |
| Iteration cap | `max_iter` | |
| Tolerance of an equality check (not a solver) | `atol` | |
| Payments or coupons per year | `freq` | `frequency` |
| Coverage of a confidence interval | `confidence` | `ci_level` |
| Random source (`Generator`, seed or `None`) | `rng` | `seed`, `random_state` |

- `tol` documents whether it is absolute or relative.
- `n_iter` is a fixed iteration count (`normalised_implied_volatility`);
  `max_iter` caps a loop that stops at `tol`.
- `n_replications` (repeats of a whole estimator, to measure its error)
  and `n_samples` (draws in one estimate) are distinct from `n_paths`.
- `nlags` is the highest lag a function returns (`autocorrelation`);
  `lags` is the lag order of a model or test (`adf_test`, `newey_west`).
- A PDE grid keeps `n_space` and `n_time` rather than `n_steps`, because
  it has two axes.

**Result fields.** The same quantity has the same name in every result:

| Quantity | Field |
|---|---|
| p-value | `p_value` (`<test>_p_value` when there are several) |
| Sample size | `n_obs` |
| Iterations a solver used | `n_iter` |
| Whether an optimizer converged | `converged` |
| Standard error of the result's own estimate | `std_error` (`std_errors` for an array) |
| Standard error of a named quantity | `<name>_se` (`price_se`, `alpha_se`) |
| Monte Carlo point estimate | `estimate` |
| Log-likelihood | `loglik` |

**Visualizers.** Every domain re-exports its `visualizers` from the
domain package, so `fk.<domain>.plot_*` always works and appears in the
API reference. `fk.<domain>.visualizers.plot_*` also keeps working.

## API stability and deprecation policy

financialkit is in **beta** (`Development Status :: 4 - Beta`) and
follows [Semantic Versioning](https://semver.org/). The public API is
every name listed in a subpackage's `__all__`, reached as
`fk.<subpackage>.<name>` or `financialkit.<subpackage>.<name>`,
including the fields of result dataclasses. Modules and names whose
path contains a leading underscore (for example
`financialkit._jit`), and anything under `core/`, `systems/`, or
`utils/` that isn't re-exported by its subpackage, are internal.

- **Patch releases (0.x.Y)** fix bugs only. They never change a public
  signature or a result field.
- **Minor releases (0.X.0)** may add to the public API. A public name,
  parameter, or result field is removed or renamed only after a
  deprecation period: for at least one minor release it keeps working
  and emits a `DeprecationWarning` that names its replacement, and the
  change is listed under "Deprecated" in `CHANGELOG.md`.
  `financialkit._deprecation` has one helper for each kind of rename:
  `deprecated_alias` for a function, `renamed_parameter` for a keyword
  (the parameter keeps its position, so positional calls never break)
  and `deprecated_field` for a result field, which keeps the old name
  readable. Results are built by financialkit, so a renamed field's old
  name is not accepted when constructing the dataclass yourself.
- **Numerical results** may change within tolerance in any release when
  an algorithm is made more accurate or robust. Changes beyond round-off
  are noted in the changelog.
- **1.0** will freeze the public API. After that, removals happen only in
  major releases.

## Releasing

1. Bump `__version__` in `financialkit/__init__.py` (`pyproject.toml`
   reads the version from there) and `version`/`date-released` in
   `CITATION.cff`, and move the `CHANGELOG.md` "Unreleased" entries under
   a new `## [X.Y.Z] - YYYY-MM-DD` section. Regenerate the README figures
   (`python docs/make_readme_figure.py`,
   `python docs/make_readme_subpackage_figures.py`) if their code changed.
2. Merge to `main`, then push a tag `vX.Y.Z` on that commit:

   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

   That is the only manual step. `.github/workflows/release.yml` checks
   that the tag matches the built distributions, `CITATION.cff` and a
   CHANGELOG section; uploads to PyPI through trusted publishing (no API
   token); creates the GitHub Release with the CHANGELOG section as its
   notes (Zenodo then archives it with a DOI); and deploys the docs to
   `gh-pages`. The one-time setup on PyPI, GitHub and Zenodo is described
   in the workflow's header comment.

## History entries

`docs/source/history/*_breakthroughs.rst` is a curated chronology per
domain, not a general-purpose list of "interesting finance history." Each
entry exists because it connects to something this package actually
implements. Keep it that way:

- A new entry must cite the concrete class/method it connects to
  (`:meth:`.../:class:`...` cross-reference) and belongs in the same PR
  as the implementation it describes, not added on its own.
- One entry per PR. If you're adding several related models,
  open separate PRs (or ask first) rather than batching a set of history
  entries together.
- Changes to `docs/source/history/**` require a maintainer review
  (enforced via `.github/CODEOWNERS`) even if the rest of the PR is
  otherwise approved.

## Reporting bugs / requesting features

Open a GitHub issue. For a pricing or model bug specifically, include
the formula or reference you expected the code to match, and (if
possible) the numeric discrepancy — "this doesn't look right" is much
slower to act on than "Hull's Example 15.6 gives a call price of 4.76
and I'm getting X."

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
