Benchmarks
==========

financialkit runs with or without `numba <https://numba.pydata.org/>`_.
Without it, :func:`financialkit.integrators.njit` is a no-op and every
``@njit`` kernel runs as plain Python: the same results, only slower.
This table puts numbers on "slower". It was produced by
``benchmarks/bench_njit.py``, which runs each workload with numba and then
in a separate process where ``import numba`` fails (as under Pyodide and
JupyterLite), checks that both give the same numbers, and reports the best
of 5 calls after a warm-up call.

.. list-table::
   :header-rows: 1
   :widths: 34 18 10 10 8 10 10

   * - Workload
     - ``@njit`` kernel
     - With numba (ms)
     - Without numba (ms)
     - Speed-up
     - First call with numba (ms)
     - Max. rel. difference
   * - CRR tree, American put, 2,000 steps
     - None (vectorised NumPy)
     - 28.7
     - 28.6
     - 1.0x
     - 30
     - 0
   * - Trinomial tree, American put, 1,000 steps
     - None (vectorised NumPy)
     - 13.8
     - 13.8
     - 1.0x
     - 14
     - 0
   * - Crank-Nicolson American put by PSOR, 200 x 200 grid
     - ``psor_kernel``
     - 4.5
     - 164.5
     - 36.5x
     - 245
     - 0
   * - Crank-Nicolson American put by Brennan-Schwartz, 200 x 200 grid
     - ``brennan_schwartz_kernel``
     - 2.5
     - 22.9
     - 9.2x
     - 101
     - 0
   * - Longstaff-Schwartz American put, 20,000 paths, 50 dates
     - None (vectorised NumPy)
     - 26.2
     - 25.7
     - 1.0x
     - 35
     - 0
   * - Euler-Maruyama, GBM, 10,000 paths x 252 steps
     - None (vectorised NumPy)
     - 5.5
     - 5.2
     - 1.0x
     - 13
     - 0
   * - Milstein, GBM, 10,000 paths x 252 steps
     - None (vectorised NumPy)
     - 9.4
     - 9.1
     - 1.0x
     - 10
     - 0
   * - Euler-Maruyama, Heston (full truncation), 10,000 paths x 252 steps
     - None (vectorised NumPy)
     - 58.3
     - 56.2
     - 1.0x
     - 73
     - 0

Measured on Darwin 25.6.0 on arm, Python 3.14.7, NumPy 2.5.3, numba 0.68.0; 2026-10-21. Times depend on the machine; the ratios are the
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
