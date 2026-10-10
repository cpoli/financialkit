"""Sphinx configuration for financialkit."""

import os
import re
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

sys.path.insert(0, os.path.abspath("../.."))


def _release():
    """Installed distribution version, falling back to financialkit/__init__.py.

    The metadata lookup only succeeds when financialkit is installed (or a
    stale, gitignored egg-info happens to sit in the repo root), so a docs
    build from a fresh clone or a docs-only environment would otherwise die
    with PackageNotFoundError.
    """
    try:
        return _dist_version("financialkit")
    except PackageNotFoundError:
        init = Path(__file__).resolve().parents[2] / "financialkit" / "__init__.py"
        return re.search(r'^__version__\s*=\s*"([^"]+)"', init.read_text(), re.M).group(1)


project = "financialkit"
copyright = "2026, financialkit contributors"
author = "financialkit team"
release = _release()
del _release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Supports NumPy-style docstrings
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "jupyterlite_sphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Single source of truth for the finance subpackages. Every card grid
# (homepage, api/index, examples/index, history/index, and the per-subpackage
# hub pages) and the cross-link strip atop each api/examples/history page is
# generated from this table by _generate_subpackage_docs() below, instead of
# being hand-duplicated across five-plus RST files that used to drift out of
# sync with each other (different card ordering, stale blurbs, etc.) --
# ported structure-for-structure from physicskit's conf.py.
SUBPACKAGES = [
    {
        "name": "time_value",
        "category": "Foundations",
        "blurb": "Compounding and equivalent rates, present value and NPV, annuities, IRR and MIRR, amortization and the APR, and day-count conventions.",
    },
    {
        "name": "fixed_income",
        "category": "Foundations",
        "blurb": "Bond prices and yields, STRIPS, Macaulay, modified, Fisher-Weil and key rate durations, convexity, bootstrapped, spline, Nelson-Siegel and Svensson curves, immunization, and swaps.",
    },
    {
        "name": "options",
        "category": "Derivatives",
        "blurb": "Bachelier, Black-Scholes-Merton, Black-76 and Garman-Kohlhagen, parity and bounds, the Greeks and delta hedging, implied volatility and the smile, and binomial and trinomial trees.",
    },
    {
        "name": "exotic_options",
        "category": "Derivatives",
        "blurb": "Merton, Reiner-Rubinstein, double and discretely monitored barriers, binaries and one-touches, Kemna-Vorst and Turnbull-Wakeman Asians, lookbacks, variance swaps, Geske compounds, forward starts and choosers, Margrabe, Stulz and quanto options, and Longstaff-Schwartz.",
    },
    {
        "name": "stochastic_volatility",
        "category": "Derivatives",
        "blurb": "Cox's CEV and Dupire local volatility, Hull-White and Heston stochastic volatility, Merton, variance gamma, Kou and Bates jumps, Carr-Madan FFT, Lewis and COS pricing, SABR, SVI and Lee's moment formula, and full-truncation and QE simulation.",
    },
    {
        "name": "monte_carlo",
        "category": "Numerical Methods",
        "blurb": "Crude Monte Carlo with error bars, stratification, importance sampling, antithetic, control and moment-matched variates, Latin hypercubes, Halton and Sobol QMC, likelihood-ratio and pathwise Greeks, and multilevel Monte Carlo.",
    },
    {
        "name": "pde_methods",
        "category": "Numerical Methods",
        "blurb": "Explicit, implicit and Crank-Nicolson schemes with Rannacher smoothing, the Thomas algorithm, CFL and von Neumann stability, Richardson extrapolation and Tavella-Randall grids, the heat-equation transform and trinomial equivalence, barriers, American options by PSOR, Brennan-Schwartz and penalty, and Peaceman-Rachford ADI.",
    },
    {
        "name": "interest_rate_models",
        "category": "Rates & Credit",
        "blurb": "Merton, Vasicek, CIR and affine short rates, Ho-Lee, Hull-White, G2++ and Black-Derman-Toy fitted to the curve, Jamshidian bond options and swaptions, Black caps and swaptions, and HJM.",
    },
    {
        "name": "credit",
        "category": "Rates & Credit",
        "blurb": "Altman's Z-score, Merton, Black-Cox and Leland structural models and the KMV distance to default, hazard curves and recovery conventions, CDS pricing and bootstrapping, CreditMetrics and generator matrices, Vasicek's large pool, CreditRisk+, Li's copula, the Basel IRB formula and CDO tranches.",
    },
    {
        "name": "asset_pricing",
        "category": "Portfolios & Asset Pricing",
        "blurb": "Williams's dividend discount model, Gordon growth, Modigliani-Miller and Shiller's variance bound, the CAPM, Jensen's alpha, the zero-beta CAPM and the GRS test, Fama-MacBeth, the APT, Fama-French and Carhart factors, Lucas's tree, the consumption CAPM and the equity premium puzzle, state prices and the Hansen-Jagannathan bound.",
    },
    {
        "name": "portfolio",
        "category": "Portfolios & Asset Pricing",
        "blurb": "Markowitz, Tobin and Merton frontiers, Kelly, Merton's portfolio problem and CPPI, single-index and Ledoit-Wolf covariances, Black-Litterman and resampling, CVaR optimization, maximum diversification, risk parity and HRP, and the Sharpe, Sortino and information ratios.",
    },
    {
        "name": "risk",
        "category": "Risk & Markets",
        "blurb": "Historical, parametric, Cornish-Fisher, Monte Carlo, hybrid and filtered VaR, safety first, expected shortfall and coherence, Euler allocation, extreme value theory, VaR backtests, copulas and tail dependence, and drawdowns.",
    },
    {
        "name": "time_series",
        "category": "Risk & Markets",
        "blurb": "Fat tails and the Hill index, random-walk tests (runs, Ljung-Box, variance ratios), Yule-Walker and ARMA with AIC, Dickey-Fuller and Engle-Granger, the Hurst exponent, ARCH, GARCH, EGARCH, GJR and EWMA, and range and realized volatility.",
    },
    {
        "name": "market_microstructure",
        "category": "Risk & Markets",
        "blurb": "Quoted, effective and realized spreads, the Roll and Corwin-Schultz estimators, Lee-Ready classification, Garman, Ho-Stoll and Avellaneda-Stoikov dealers, Kyle and Glosten-Milgrom, Hasbrouck's VAR, PIN and VPIN, square-root impact, Bertsimas-Lo and Almgren-Chriss execution, and Amihud illiquidity.",
    },
]

for _s in SUBPACKAGES:
    _s.setdefault("history_doc", f"{_s['name']}_breakthroughs")
    # Every subpackage's "Examples" link goes straight to its sphinx-gallery
    # index (:orphan: like every such index -- meant to be linked to
    # directly rather than placed in a toctree). Where a narrative tutorial
    # also exists, that gallery's examples/<name>/README.rst header links
    # out to it, so there's no separate examples/<name>.rst stub page whose
    # only job is forwarding to one or the other.
    _s["examples_doc"] = f"api/gallery/{_s['name']}/index"
del _s

_CATEGORY_ORDER = [
    "Foundations",
    "Derivatives",
    "Numerical Methods",
    "Rates & Credit",
    "Portfolios & Asset Pricing",
    "Risk & Markets",
]

# Subpackages with a sphinx-gallery-formatted examples/<name>/ directory.
_GALLERY_SUBPACKAGES = [s["name"] for s in SUBPACKAGES]


def _jupyterlite_install_cell(notebook_content, notebook_filename):
    """Prepend a ``%pip install`` cell so the Pyodide kernel fetches financialkit from PyPI."""
    notebook_content["cells"].insert(
        0,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%pip install financialkit"],
        },
    )


sphinx_gallery_conf = {
    "examples_dirs": [f"../../examples/{name}" for name in _GALLERY_SUBPACKAGES],
    "gallery_dirs": [f"api/gallery/{name}" for name in _GALLERY_SUBPACKAGES],
    "filename_pattern": r"/plot_",
    # Zip downloads of each gallery's .py/.ipynb files, so a course can take a
    # whole domain's examples at once.
    "download_all_examples": True,
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
    # Lets ".. minigallery::" (used throughout docs/source/history/) resolve
    # fully-qualified object names in addition to the file paths/globs it
    # already handles, and is required for it to not warn about falling
    # back to file-path resolution on every single invocation.
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("financialkit",),
    # Embed each FuncAnimation as matplotlib's HTML/JavaScript player
    # (to_jshtml): PNG frames, no ffmpeg, the same player Jupyter and
    # JupyterLite show for `plt.rcParams["animation.html"] = "jshtml"`.
    "matplotlib_animations": (True, "jshtml"),
    # The examples end a cell with a bare `anim` so that notebooks display
    # the player; the scraper above already embeds it in the gallery page,
    # so don't also capture its repr.
    "ignore_repr_types": r"matplotlib\.animation\.",
    # "Launch JupyterLite" button: runs the notebook in the browser on
    # Pyodide, with no server. Numba cannot run there, so this relies on
    # numba being optional (financialkit/_jit.py).
    "jupyterlite": {
        "use_jupyter_lab": True,
        "notebook_modification_function": _jupyterlite_install_cell,
    },
}

# Many domains share common attribute names (e.g. "x", "n") across unrelated
# classes; autolinking to a single target for them isn't meaningful, so
# treat those as non-fatal rather than broken links.
suppress_warnings = ["ref.python", "config.cache"]

html_theme = "pydata_sphinx_theme"
html_logo = "_static/images/financialkit_logo_transparent.png"
html_theme_options = {
    "github_url": "https://github.com/cpoli/financialkit",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/financialkit/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "navigation_depth": 2,
    "logo": {
        "alt_text": "financialkit logo",
    },
}
html_static_path = ["_static"]


def _card(link, blurb, link_title):
    """One sphinx-design grid-item-card, indented for direct concatenation into a ``.. grid::`` block."""
    return f"   .. grid-item-card:: {link_title}\n      :link: {link}\n      :link-type: doc\n\n      {blurb}\n\n"


def _grid(cards):
    return ".. grid:: 1 2 3 3\n   :gutter: 2\n\n" + "".join(cards)


def _grouped_grid(link_fn):
    """A ``.. grid::`` per category (in _CATEGORY_ORDER), each preceded by a
    rubric heading, with ``link_fn(subpackage)`` giving each card's target.
    Shared by the homepage hub grid and the api/examples/history grids so
    they show the same category grouping instead of a flat alphabetical
    list.
    """
    parts = []
    for category in _CATEGORY_ORDER:
        members = [s for s in SUBPACKAGES if s["category"] == category]
        if not members:
            continue
        parts.append(f".. rubric:: {category}\n\n")
        parts.append(_grid([_card(link_fn(s), s["blurb"], f"financialkit.{s['name']}") for s in members]))
        parts.append("\n")
    return "".join(parts)


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _generate_subpackage_docs(app):
    """Generate the card grids, per-subpackage hub pages, and cross-link
    strips derived from SUBPACKAGES.

    Runs at "builder-inited", the same point sphinx-gallery uses to write
    its own generated RST, so the output exists before Sphinx reads any
    source file that ``.. include::``/``.. toctree::``\\ s it. Everything
    lands under _generated/ (gitignored, like api/gallery/) rather than
    being committed, since it's entirely derived from SUBPACKAGES above --
    there's nothing hand-authored in it to keep in version control.
    """
    out = Path(app.srcdir) / "_generated"

    # RST substitutions for numbers/facts derived from SUBPACKAGES, so prose
    # elsewhere (e.g. the homepage's "N domains" pitch) doesn't hardcode a
    # count that silently goes stale the next time a subpackage is added.
    _write(out / "vars.rst", f".. |num_subpackages| replace:: {len(SUBPACKAGES)}\n")

    # Category-grouped grids reused by index.rst, api/index.rst,
    # examples/index.rst, and history/index.rst, so those five-plus listings
    # of the same subpackages -- and their grouping -- come from one
    # place instead of being hand-copied (and drifting) independently.
    _write(out / "grid_api.rst", _grouped_grid(lambda s: f"/api/{s['name']}"))
    _write(out / "grid_examples.rst", _grouped_grid(lambda s: f"/{s['examples_doc']}"))
    _write(out / "grid_history.rst", _grouped_grid(lambda s: f"/history/{s['history_doc']}"))

    # Domain-first homepage grid: same grouping, linking to each
    # subpackage's own hub page rather than straight into a single
    # History/Examples/API silo.
    _write(out / "grid_hub.rst", _grouped_grid(lambda s: f"/_generated/subpackages/{s['name']}"))

    # Per-subpackage hub page: History, Examples, and API reference as
    # three equally-weighted doors into the same domain, so a reader who
    # wants "everything about linalg" doesn't have to browse three
    # separate site-wide sections to find linalg in each of them.
    hub_dir = out / "subpackages"
    for s in SUBPACKAGES:
        name = s["name"]
        title = f"financialkit.{name}"
        cards = _grid(
            [
                _card(
                    f"/history/{s['history_doc']}",
                    "The foundational breakthroughs behind this subpackage, linked to the implementation.",
                    "History",
                ),
                _card(
                    f"/{s['examples_doc']}",
                    "Runnable tutorials and the full example gallery.",
                    "Examples",
                ),
                _card(
                    f"/api/{name}",
                    "Every public class and function.",
                    "API reference",
                ),
            ]
        )
        _write(hub_dir / f"{name}.rst", f"{title}\n{'=' * len(title)}\n\n{s['blurb']}\n\n{cards}")

    # Cross-link strip included atop each hand-authored api/<name>.rst and
    # history/<name>_breakthroughs.rst page, so a reader who lands on just
    # one of them (e.g. from a search engine) can discover the others for
    # the same subpackage.
    nav_dir = out / "nav"
    for s in SUBPACKAGES:
        name = s["name"]
        links = [
            f":doc:`financialkit.{name} hub </_generated/subpackages/{name}>`",
            f":doc:`History </history/{s['history_doc']}>`",
            f":doc:`Examples </{s['examples_doc']}>`",
            f":doc:`API reference </api/{name}>`",
        ]
        _write(nav_dir / f"{name}.rst", f".. container:: subpkg-nav\n\n   {' · '.join(links)}\n")


def setup(app):
    app.connect("builder-inited", _generate_subpackage_docs)
