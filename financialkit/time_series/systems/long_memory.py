"""Long memory: Hurst's rescaled range and fractional Gaussian noise.

Designing reservoirs on the Nile, Hurst (1951) found that the range of
cumulative departures from the mean, divided by the standard deviation,
grows like :math:`n^H` with :math:`H \\approx 0.73` rather than the
:math:`n^{1/2}` of independent inflows. Mandelbrot and Van Ness (1968)
explained the exponent with fractional Brownian motion, whose increments
(fractional Gaussian noise) have autocovariance
:math:`\\gamma(k) = \\tfrac12(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})`. In
finance, :math:`H > 1/2` would mean persistent, predictable returns. Anis
and Lloyd (1976) gave the exact small-sample expectation of R/S for
independent normals, against which the estimate is corrected. References:
H. E. Hurst, "Long-Term Storage Capacity of Reservoirs," Transactions of
the American Society of Civil Engineers 116 (1951), 770-799; B. B.
Mandelbrot and J. W. Van Ness, SIAM Review 10(4) (1968), 422-437; A. A.
Anis and E. H. Lloyd, Biometrika 63(1) (1976), 111-116; R. B. Davies and D. S.
Harte, Biometrika 74(1) (1987), 95-101.
"""

from __future__ import annotations

import numpy as np
from scipy import fft
from scipy.special import gammaln

from financialkit.time_series.core.base import HurstResult

__all__ = ["rescaled_range", "anis_lloyd_expected_rs", "hurst_exponent", "fgn_autocovariance", "simulate_fgn"]


def rescaled_range(x) -> float:
    """Hurst's adjusted rescaled range :math:`R/S` of one block.

    :math:`R` is the range of the cumulative sums of deviations from the
    block mean and :math:`S` the block standard deviation.

    Examples
    --------
    >>> rescaled_range([1.0, -1.0, 1.0, -1.0])
    1.0
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.cumsum(x - x.mean())
    return float((max(y.max(), 0.0) - min(y.min(), 0.0)) / x.std())


def anis_lloyd_expected_rs(n) -> np.ndarray:
    """Expected :math:`R/S` of `n` independent normals (Anis and Lloyd 1976).

    .. math::

       E[(R/S)_n] = \\frac{\\Gamma(\\frac{n-1}{2})}{\\sqrt\\pi\\,\\Gamma(\\frac n2)}
       \\sum_{i=1}^{n-1}\\sqrt{\\frac{n-i}{i}}

    Examples
    --------
    >>> import numpy as np
    >>> float(np.round(anis_lloyd_expected_rs(10), 4))
    3.0233
    """
    ns = np.atleast_1d(np.asarray(n, dtype=np.int64))
    out = np.empty(ns.size)
    for k, m in enumerate(ns):
        if m < 3:
            raise ValueError("n must be at least 3")
        i = np.arange(1, m)
        s = np.sum(np.sqrt((m - i) / i))
        out[k] = np.exp(gammaln((m - 1) / 2) - gammaln(m / 2)) / np.sqrt(np.pi) * s
    return out if np.ndim(n) else out[0]


def hurst_exponent(x, min_window: int = 16, n_windows: int = 12) -> HurstResult:
    """Estimate Hurst's exponent by rescaled-range (R/S) analysis.

    For each window length :math:`n` (log-spaced from `min_window` to half
    the sample), the series is cut into non-overlapping blocks and the
    block values of :math:`R/S` are averaged. :math:`H` is the slope of
    :math:`\\log (R/S)_n` on :math:`\\log n`. Because :math:`(R/S)_n` of
    independent data approaches :math:`\\sqrt{\\pi n/2}` only slowly, the
    raw slope exceeds 1/2 in small windows; :attr:`HurstResult.hurst_corrected`
    regresses the excess over the Anis-Lloyd expectation instead.

    Parameters
    ----------
    x : array_like
        Increments (returns), not levels.
    min_window : int
    n_windows : int

    Returns
    -------
    HurstResult

    Examples
    --------
    >>> res = hurst_exponent(simulate_fgn(8192, 0.8, rng=0))
    >>> bool(0.7 < res.hurst_corrected < 0.9)
    True
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if N < 4 * min_window:
        raise ValueError("series too short for the minimum window")
    windows = np.unique(np.logspace(np.log10(min_window), np.log10(N // 2), n_windows).astype(np.int64))
    rs = np.empty(windows.size)
    for k, n in enumerate(windows):
        blocks = x[: (N // n) * n].reshape(-1, n)
        y = np.cumsum(blocks - blocks.mean(axis=1, keepdims=True), axis=1)
        R = np.maximum(y.max(axis=1), 0.0) - np.minimum(y.min(axis=1), 0.0)
        rs[k] = np.mean(R / blocks.std(axis=1))
    expected = anis_lloyd_expected_rs(windows)
    logn = np.log(windows)
    H = float(np.polyfit(logn, np.log(rs), 1)[0])
    H_corr = 0.5 + float(np.polyfit(logn, np.log(rs) - np.log(expected), 1)[0])
    return HurstResult(window=windows, rs=rs, expected_rs=expected, hurst=H, hurst_corrected=H_corr)


def fgn_autocovariance(H: float, nlags: int) -> np.ndarray:
    """Autocovariances ``k = 0..nlags`` of unit-variance fractional Gaussian noise.

    Examples
    --------
    >>> fgn_autocovariance(0.5, 3).tolist()
    [1.0, 0.0, 0.0, 0.0]
    """
    if not 0 < H < 1:
        raise ValueError("H must lie in (0, 1)")
    k = np.arange(nlags + 1, dtype=np.float64)
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def simulate_fgn(n: int, H: float, sigma: float = 1.0, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Exact fractional Gaussian noise by circulant embedding (Davies and Harte 1987).

    The autocovariance is embedded in a circulant matrix of size :math:`2n`,
    whose eigenvalues are its FFT (:func:`scipy.fft.fft`) and are
    nonnegative for fractional Gaussian noise; colouring complex white noise
    by their square roots and transforming back gives a sample with exactly
    the target covariance. ``np.cumsum`` of the result is fractional
    Brownian motion.

    Parameters
    ----------
    n : int
    H : float
        Hurst exponent in (0, 1); 1/2 gives white noise.
    sigma : float
        Standard deviation of each increment.
    rng : numpy.random.Generator, int or None

    Returns
    -------
    ndarray, shape (n,)
    """
    rng = np.random.default_rng(rng)
    g = fgn_autocovariance(H, n)
    c = np.r_[g, g[-2:0:-1]]
    M = c.size
    lam = np.maximum(fft.fft(c).real, 0.0)
    w = np.sqrt(lam / M) * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
    return sigma * fft.fft(w).real[:n]
