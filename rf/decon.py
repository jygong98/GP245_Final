"""Receiver-function deconvolution from radial and vertical seismograms."""

from __future__ import annotations

import numpy as np

try:
    from seispy.decon import deconit, deconwater
except ImportError:
    raise ImportError(
        "seispy is required for receiver function computation.\n"
        "Install with: pip install python-seispy"
    )


def _deconvolve(
    ux: np.ndarray,
    uz: np.ndarray,
    dt: float,
    *,
    phase: str,
    tshift: float,
    f0: float,
    method: str,
    itmax: int,
    minderr: float,
    wlevel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = phase.upper()
    if phase not in ("P", "S"):
        raise ValueError("phase must be 'P' or 'S'")

    nrcv, _nt = ux.shape
    rfs = []
    rms_arr = []

    for i in range(nrcv):
        radial = ux[i].astype(np.float64)
        vertical = uz[i].astype(np.float64)
        if phase == "P":
            uin, win = radial, vertical
        else:
            uin, win = vertical, radial
        if method == "iter":
            rf, rms, _nit = deconit(
                uin, win, dt, tshift=tshift, f0=f0,
                itmax=itmax, minderr=minderr, phase=phase,
            )
        else:
            rf, rms = deconwater(
                uin, win, dt, tshift=tshift, f0=f0,
                wlevel=wlevel, phase=phase,
            )
        rfs.append(rf)
        rms_arr.append(float(np.asarray(rms).ravel()[0]))

    rfs = np.array(rfs)
    npts_rf = rfs.shape[1]
    t_rf = np.arange(npts_rf) * dt - tshift
    return rfs, t_rf, np.array(rms_arr)


def compute_prf(
    ux: np.ndarray,
    uz: np.ndarray,
    dt: float,
    *,
    tshift: float = 10.0,
    f0: float = 8.0,
    method: str = "iter",
    itmax: int = 400,
    minderr: float = 0.001,
    wlevel: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute P-wave receiver functions for all traces.

    Deconvolves radial (ux) from vertical (uz) using seispy iterative
    time-domain deconvolution (Ligorria & Ammon, 1999) or a water-level
    spectral method.

    Parameters
    ----------
    ux : array, shape (nrcv, nt)
        Radial (in-plane horizontal) displacement.
    uz : array, shape (nrcv, nt)
        Vertical displacement.
    dt : float
        Sample interval (s).
    tshift : float
        Time shift before the direct arrival (s).
    f0 : float
        Gaussian filter width factor.
    method : str
        Deconvolution method: ``'iter'`` or ``'water'``.
    itmax : int
        Maximum iterations (iterative method only).
    minderr : float
        Minimum error improvement (iterative method only).
    wlevel : float
        Water level (water-level method only).

    Returns
    -------
    rfs : array, shape (nrcv, npts_rf)
        P-wave receiver function time series per trace.
    t_rf : array, shape (npts_rf,)
        Time axis for the RF (starting at ``-tshift``).
    rms_arr : array, shape (nrcv,)
        RMS misfit for each trace deconvolution.
    """
    return _deconvolve(
        ux, uz, dt,
        phase="P", tshift=tshift, f0=f0, method=method,
        itmax=itmax, minderr=minderr, wlevel=wlevel,
    )


def compute_srf(
    ux: np.ndarray,
    uz: np.ndarray,
    dt: float,
    *,
    tshift: float = 6.5,
    f0: float = 8.0,
    method: str = "iter",
    itmax: int = 400,
    minderr: float = 0.001,
    wlevel: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute S-wave receiver functions for all traces.

    Deconvolves vertical (uz) from radial (ux), matching FDFK2D
    ``plot_SRF.m``.
    """
    return _deconvolve(
        ux, uz, dt,
        phase="S", tshift=tshift, f0=f0, method=method,
        itmax=itmax, minderr=minderr, wlevel=wlevel,
    )
