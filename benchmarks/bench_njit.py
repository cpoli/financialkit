"""Time financialkit's numerical hot spots with and without numba.

Each case runs in two child processes: one with numba, and one where
``import numba`` fails, so :data:`financialkit._jit.HAS_NUMBA` is false and
every ``@njit`` kernel runs as plain Python, exactly as under Pyodide. The
parent checks that both give the same numbers and prints a table; with
``--rst`` it also rewrites the documentation page.

Timings are the best of ``--repeat`` calls after one warm-up call, so
numba's compilation is excluded from them; the first call is reported in
its own column.

Usage::

    pip install -e ".[fast]"
    python benchmarks/bench_njit.py                                  # print the table
    python benchmarks/bench_njit.py --rst docs/source/benchmarks.rst  # and rewrite the docs page
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2


def _cases() -> list[tuple[str, str, str, Callable[[], np.ndarray]]]:
    """``(name, kernel, workload, run)``; ``run`` returns the numbers to compare."""
    import financialkit as fk
    from financialkit.monte_carlo.utils.numerics import gbm_diffusion, gbm_drift

    def gbm_dx(x, t, p):
        return np.full_like(x, p[1])

    params = np.array([r, sigma])
    dW = np.random.default_rng(0).standard_normal((10_000, 252)) * np.sqrt(T / 252)
    heston = fk.stochastic_volatility.HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

    def american_fd(method: str) -> np.ndarray:
        res = fk.pde_methods.CrankNicolson(200, 200, american_method=method).price(S, K, T, r, sigma, option_type="put", american=True)
        return np.array([res.price])

    def tree(pricer, n: int) -> np.ndarray:
        return np.array([pricer(n).price(S, K, T, r, sigma, option_type="put", american=True).price])

    return [
        ("crr", "None (vectorised NumPy)", "CRR tree, American put, 2,000 steps", lambda: tree(fk.options.CRRTree, 2000)),
        ("trinomial", "None (vectorised NumPy)", "Trinomial tree, American put, 1,000 steps", lambda: tree(fk.options.TrinomialTree, 1000)),
        ("psor", "``psor_kernel``", "Crank-Nicolson American put by PSOR, 200 x 200 grid", lambda: american_fd("psor")),
        (
            "brennan_schwartz",
            "``brennan_schwartz_kernel``",
            "Crank-Nicolson American put by Brennan-Schwartz, 200 x 200 grid",
            lambda: american_fd("brennan_schwartz"),
        ),
        (
            "lsm",
            "None (vectorised NumPy)",
            "Longstaff-Schwartz American put, 20,000 paths, 50 dates",
            lambda: np.array([fk.exotic_options.longstaff_schwartz(S, K, T, r, sigma, n_paths=20_000, rng=0).estimate]),
        ),
        (
            "euler_gbm",
            "None (vectorised NumPy)",
            "Euler-Maruyama, GBM, 10,000 paths x 252 steps",
            lambda: fk.integrators.euler_maruyama(gbm_drift, gbm_diffusion, S, T, 252, 10_000, params, dW=dW)[1][:, -1],
        ),
        (
            "milstein_gbm",
            "None (vectorised NumPy)",
            "Milstein, GBM, 10,000 paths x 252 steps",
            lambda: fk.integrators.milstein(gbm_drift, gbm_diffusion, gbm_dx, S, T, 252, 10_000, params, dW=dW)[1][:, -1],
        ),
        (
            "euler_heston",
            "None (vectorised NumPy)",
            "Euler-Maruyama, Heston (full truncation), 10,000 paths x 252 steps",
            lambda: fk.stochastic_volatility.heston_paths(heston, S, T, r, n_steps=252, n_paths=10_000, scheme="full_truncation", rng=0).spot[:, -1],
        ),
    ]


def _child(out_dir: Path, repeat: int, no_numba: bool) -> None:
    if no_numba:
        sys.modules["numba"] = None  # type: ignore[assignment]  # makes `import numba` raise ImportError
    from financialkit._jit import HAS_NUMBA

    assert HAS_NUMBA != no_numba, "numba must be installed to benchmark the JIT"
    report = {}
    for name, _, _, run in _cases():
        t0 = time.perf_counter()
        values = run()
        first = time.perf_counter() - t0
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            run()
            times.append(time.perf_counter() - t0)
        np.save(out_dir / f"{name}.npy", values)
        report[name] = {"first": first, "best": min(times)}
    (out_dir / "times.json").write_text(json.dumps(report))


def _run_child(repeat: int, no_numba: bool) -> tuple[dict, Path]:
    out_dir = Path(tempfile.mkdtemp(prefix="fk-bench-"))
    cmd = [sys.executable, __file__, "--child", str(out_dir), "--repeat", str(repeat)] + (["--no-numba"] if no_numba else [])
    subprocess.run(cmd, check=True, env={**os.environ, "MPLBACKEND": "Agg"})
    return json.loads((out_dir / "times.json").read_text()), out_dir


def _rows(repeat: int) -> list[list[str]]:
    jit, jit_dir = _run_child(repeat, no_numba=False)
    py, py_dir = _run_child(repeat, no_numba=True)
    rows = []
    for name, kernel, workload, _ in _cases():
        a, b = np.load(jit_dir / f"{name}.npy"), np.load(py_dir / f"{name}.npy")
        diff = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
        if diff > 1e-10:
            raise AssertionError(f"{name}: numba and plain Python disagree (max relative difference {diff:.1e})")
        t_jit, t_py = jit[name]["best"], py[name]["best"]
        rows.append(
            [
                workload,
                kernel,
                f"{1e3 * t_jit:,.1f}",
                f"{1e3 * t_py:,.1f}",
                f"{t_py / t_jit:,.1f}x",
                f"{1e3 * jit[name]['first']:,.0f}",
                f"{diff:.0e}" if diff else "0",
            ]
        )
    return rows


HEADER = ["Workload", "``@njit`` kernel", "With numba (ms)", "Without numba (ms)", "Speed-up", "First call with numba (ms)", "Max. rel. difference"]


def _rst_table(rows: list[list[str]]) -> str:
    lines = [".. list-table::", "   :header-rows: 1", "   :widths: 34 18 10 10 8 10 10", ""]
    for row in [HEADER, *rows]:
        lines.append(f"   * - {row[0]}")
        lines.extend(f"     - {cell}" for cell in row[1:])
    return "\n".join(lines)


def _environment() -> str:
    import numba

    cpu = platform.processor() or platform.machine()
    return (
        f"{platform.system()} {platform.release()} on {cpu}, Python {platform.python_version()}, "
        f"NumPy {np.__version__}, numba {numba.__version__}; {datetime.date.today().isoformat()}"
    )


PAGE = """\
Benchmarks
==========

financialkit runs with or without `numba <https://numba.pydata.org/>`_.
Without it, :func:`financialkit.integrators.njit` is a no-op and every
``@njit`` kernel runs as plain Python: the same results, only slower.
This table puts numbers on "slower". It was produced by
``benchmarks/bench_njit.py``, which runs each workload with numba and then
in a separate process where ``import numba`` fails (as under Pyodide and
JupyterLite), checks that both give the same numbers, and reports the best
of {repeat} calls after a warm-up call.

{table}

Measured on {environment}. Times depend on the machine; the ratios are the
part to read.

* **Speed-up** is the time without numba divided by the time with it.
* **First call with numba** includes compiling the kernel for that
  workload's argument types, which every new Python process pays once.
* **Max. rel. difference** compares the two runs' outputs (prices, or
  every path's terminal value), so it shows that numba changes the speed,
  not the results.

The binomial and trinomial trees and Longstaff-Schwartz have no ``@njit``
kernel: their loops are over time steps, with each step vectorised over
nodes or paths in NumPy, so numba makes no difference to them. The same
holds for the Euler-Maruyama and Milstein schemes, whose time loop is
vectorised over paths: they were ``@njit`` kernels until compiled runs
measured no faster (0.8-0.9x) and paid up to a second of compilation, so
they are plain Python now. PSOR and Brennan-Schwartz loop over grid
nodes one by one, which is where the JIT pays off, so numba is worth
installing for the PDE solvers.

To reproduce it::

    pip install -e ".[fast]"
    python benchmarks/bench_njit.py --rst docs/source/benchmarks.rst
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=5, help="timed calls per workload (default 5)")
    parser.add_argument("--rst", type=Path, help="rewrite this documentation page with the table")
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--no-numba", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        _child(args.child, args.repeat, args.no_numba)
        return
    rows = _rows(args.repeat)
    widths = [max(len(row[i]) for row in [HEADER, *rows]) for i in range(len(HEADER))]
    for row in [HEADER, *rows]:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)))
    if args.rst:
        args.rst.write_text(PAGE.format(repeat=args.repeat, table=_rst_table(rows), environment=_environment()))
        print(f"wrote {args.rst}")


if __name__ == "__main__":
    main()
