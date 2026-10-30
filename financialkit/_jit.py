"""Optional Numba acceleration.

Numba is an optional dependency (``pip install financialkit[fast]``).
When it is installed, :func:`njit` is :func:`numba.njit`. When it is not
(including in Pyodide/JupyterLite, where Numba cannot run), :func:`njit`
is a no-op decorator: every kernel runs as plain Python/NumPy with
identical results, only slower. Both call forms are supported --
``@njit`` and ``@njit(cache=True)`` -- so decorated code never needs to
know which is in use.

Examples
--------
>>> from financialkit.integrators import njit
>>> @njit
... def square(x):
...     return x * x
>>> square(3.0)
9.0
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["HAS_NUMBA", "njit"]

try:
    from numba import njit as _numba_njit
except ImportError:
    HAS_NUMBA = False
else:  # pragma: no cover - coverage is measured (and gated) in the CI job without numba
    HAS_NUMBA = True


def _identity_njit(*args: Any, **kwargs: Any) -> Any:
    """Stand-in for :func:`numba.njit` that returns the function unchanged."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator


njit: Callable[..., Any] = _numba_njit if HAS_NUMBA else _identity_njit
