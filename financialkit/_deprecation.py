"""Deprecated aliases for renamed public functions, parameters and result fields.

Examples
--------
>>> import warnings
>>> def new(x):
...     return 2 * x
>>> old = deprecated_alias(new, "old", "pkg.new")
>>> with warnings.catch_warnings(record=True) as w:
...     warnings.simplefilter("always")
...     old(3), w[0].category.__name__
(6, 'DeprecationWarning')
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def deprecated_alias(func: Callable[..., Any], old_name: str, new_path: str) -> Callable[..., Any]:
    """Return `func` under `old_name`, warning with :class:`DeprecationWarning` on each call.

    Parameters
    ----------
    func : callable
        The function the alias forwards to.
    old_name : str
        The alias's name, as it appears in ``__all__``.
    new_path : str
        Dotted path of `func`, named in the warning and the docstring.

    Returns
    -------
    callable
    """

    @functools.wraps(func)
    def alias(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(f"{old_name} is deprecated; use {new_path}", DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    alias.__name__ = alias.__qualname__ = old_name
    alias.__doc__ = f"Deprecated alias of :func:`{new_path}`, kept for backward compatibility."
    return alias


def renamed_parameter(old: str, new: str, func_path: str) -> Callable[[F], F]:
    """Accept keyword `old` as a deprecated spelling of parameter `new`.

    Positional calls are unaffected, since the parameter keeps its place.

    Parameters
    ----------
    old : str
        The parameter's former name.
    new : str
        Its current name in the decorated function's signature.
    func_path : str
        Dotted path of the function, named in the warning.

    Returns
    -------
    callable
        A decorator.

    Examples
    --------
    >>> import warnings
    >>> @renamed_parameter("n_sims", "n_paths", "pkg.f")
    ... def f(n_paths=1):
    ...     return n_paths
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     f(n_sims=5), w[0].category.__name__
    (5, 'DeprecationWarning')
    """

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(f"{func_path}() got both {old!r} and {new!r}; pass only {new!r}")
                warnings.warn(f"{func_path}({old}=...) is deprecated; use {new}=...", DeprecationWarning, stacklevel=2)
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorate


class deprecated_field:
    """Read-only alias of a renamed result-dataclass field, warning on each access.

    Declare it in the class body without an annotation, so the dataclass does
    not treat it as a field.

    Parameters
    ----------
    new : str
        The field's (or property's) current name.

    Examples
    --------
    >>> import warnings
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class R:
    ...     n_obs: int
    ...     nobs = deprecated_field("n_obs")
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     R(3).nobs, w[0].category.__name__
    (3, 'DeprecationWarning')
    """

    def __init__(self, new: str) -> None:
        self.new = new
        self.old = new
        self.owner = ""
        self.__doc__ = f"Deprecated alias of ``{new}``, kept for backward compatibility."

    def __set_name__(self, owner: type, name: str) -> None:
        self.old = name
        self.owner = owner.__name__

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        warnings.warn(f"{self.owner}.{self.old} is deprecated; use {self.owner}.{self.new}", DeprecationWarning, stacklevel=2)
        return getattr(obj, self.new)
