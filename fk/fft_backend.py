"""FFT helpers for the FK module.

Uses :mod:`scipy.fft` with multi-threaded workers when SciPy is available;
falls back to :mod:`numpy.fft` otherwise.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.fft import irfft as _scipy_irfft
    from scipy.fft import rfft as _scipy_rfft
    from scipy.fft import rfftfreq as _scipy_rfftfreq

    _HAS_SCIPY_FFT = True
except ImportError:
    _HAS_SCIPY_FFT = False


def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """One-sided frequency bins for length-``n`` real FFT."""
    if _HAS_SCIPY_FFT:
        return _scipy_rfftfreq(n, d=d)
    return np.fft.rfftfreq(n, d=d)


def rfft(x: np.ndarray, n: int | None = None, axis: int = -1) -> np.ndarray:
    """Real-to-complex FFT (SciPy multi-threaded when available)."""
    if _HAS_SCIPY_FFT:
        return _scipy_rfft(x, n=n, axis=axis, workers=-1)
    return np.fft.rfft(x, n=n, axis=axis)


def irfft(x: np.ndarray, n: int | None = None, axis: int = -1) -> np.ndarray:
    """Complex-to-real inverse FFT (SciPy multi-threaded when available)."""
    if _HAS_SCIPY_FFT:
        return _scipy_irfft(x, n=n, axis=axis, workers=-1)
    return np.fft.irfft(x, n=n, axis=axis)
