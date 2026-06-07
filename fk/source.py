"""Source time function and incidence configuration for the FK method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fft_backend import rfft


@dataclass
class RickerSource:
    """Ricker-wavelet plane-wave source.

    Parameters
    ----------
    f0 : float
        Dominant frequency (Hz).
    strength : float
        Amplitude scale (``ds`` in the original package), default 1.0.
    incidence : str
        ``"P"`` or ``"S"`` incident plane wave.
    t0 : float, optional
        Time of the Ricker peak (s). If ``None``, defaults to ``1.0 / f0``,
        which keeps the (acausal) wavelet essentially contained in t >= 0.
    """

    f0: float
    strength: float = 1.0
    incidence: str = "P"
    t0: float | None = None

    def __post_init__(self) -> None:
        if self.f0 <= 0:
            raise ValueError("f0 must be positive.")
        inc = self.incidence.upper()
        if inc not in ("P", "S"):
            raise ValueError("incidence must be 'P' or 'S'.")
        self.incidence = inc
        if self.t0 is None:
            self.t0 = 1.0 / self.f0

    def wavelet(self, t: np.ndarray) -> np.ndarray:
        """Ricker wavelet sampled at times ``t`` (s).

        ``r(t) = (1 - 2 (pi f0 (t - t0))^2) exp(-(pi f0 (t - t0))^2)``.
        """
        t = np.asarray(t, dtype=float)
        arg = (np.pi * self.f0 * (t - self.t0)) ** 2
        return self.strength * (1.0 - 2.0 * arg) * np.exp(-arg)

    def spectrum(self, nt: int, dt: float) -> np.ndarray:
        """One-sided (rfft) spectrum of the discretized Ricker wavelet.

        Returns an array of length ``nt // 2 + 1`` consistent with
        :func:`fk.fft_backend.rfft` on a length-``nt`` real signal sampled at ``dt``.
        """
        t = np.arange(nt) * dt
        return rfft(self.wavelet(t))
