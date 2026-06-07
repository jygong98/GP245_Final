"""Top-level FK seismogram computation.

Assembles the frequency-domain plane-wave response from a pluggable
:mod:`fk.propagators` backend, applies the source spectrum and the plane-wave
moveout across receivers, and inverse-FFTs to time-domain displacement seismograms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fft_backend import irfft, rfftfreq
from .geometry import rayp_deg_to_si
from .model import FKLayerModel
from .propagators.haskell import surface_state, state_at_depth, incident_amplitude
from .propagators.kennett import surface_state_kennett, state_at_depth_kennett
from .source import RickerSource


@dataclass
class FKResult:
    """Result of an FK seismogram computation.

    Attributes
    ----------
    t : numpy.ndarray
        Time axis (s), shape ``(nt,)``.
    ux, uz : numpy.ndarray
        Radial (in-plane horizontal) and vertical displacement seismograms,
        shape ``(nrcv, nt)``.
    receivers_x, receivers_z : numpy.ndarray
        Receiver coordinates (m).
    p_si : float
        Horizontal slowness used (s/m).
    z_init : float
        Plane-wave initialization depth used (m).
    """

    t: np.ndarray
    ux: np.ndarray
    uz: np.ndarray
    receivers_x: np.ndarray
    receivers_z: np.ndarray
    p_si: float
    z_init: float


def _analytic_ricker_spectrum(omega_c, f0, t0, strength):
    """Analytic Ricker spectrum at (possibly complex) angular frequency.

    ``R(omega) = strength * (2/sqrt(pi)) * (omega^2/omega_p^3)
                 * exp(-(omega/omega_p)^2) * exp(-i omega t0)``,
    with ``omega_p = 2 pi f0``. Used when frequency damping is enabled.
    """
    omega_p = 2.0 * np.pi * f0
    return (
        strength
        * (2.0 / np.sqrt(np.pi))
        * (omega_c ** 2 / omega_p ** 3)
        * np.exp(-((omega_c / omega_p) ** 2))
        * np.exp(-1j * omega_c * t0)
    )


def compute_fk_seismograms(
    model: FKLayerModel,
    source: RickerSource,
    receivers_x,
    dt: float,
    nt: int,
    *,
    p_si: float | None = None,
    p_deg: float | None = None,
    receivers_z=None,
    x0_plane: float = 0.0,
    z_init: float | None = None,
    phi: float = 0.0,
    model_max_depth: float | None = None,
    freq_damping: float = 0.0,
    method: str = "kennett_numba",
) -> FKResult:
    """Compute FK displacement seismograms at the given receivers.

    Parameters
    ----------
    model : FKLayerModel
        1D layered model (last entry = half-space).
    source : RickerSource
        Source time function and incidence type ('P' or 'S').
    receivers_x : array_like
        Receiver x-coordinates along the profile (m).
    dt : float
        Time sampling interval (s).
    nt : int
        Number of time samples.
    p_si, p_deg : float, optional
        Horizontal slowness in s/m or s/deg (give exactly one).
    receivers_z : array_like, optional
        Receiver depths (m); default all 0 (free surface).
    x0_plane : float
        Reference x of the plane-wave phase (m). Moveout uses ``x - x0_plane``.
    z_init : float, optional
        Vertical reference depth of the incident wavefront (m), inside the
        half-space. If ``None`` it defaults to ``model_max_depth``. This sets
        the time origin: t=0 is when the wavefront passes ``(x0_plane, z_init)``.
        (The paper's plane-wave init point ``z_o = z_init + (L/2) tan(theta)``
        used for FD coupling differs only by the constant delay ``(L/2)*p``.)
    phi : float
        Angle between the back-azimuth and the profile azimuth (rad),
        i.e. ``phi = baz - az_org``.  Same definition as Fortran's
        ``bazs = bazs - az_org*deg2rad``.  Horizontal moveout uses
        ``p_along = -p * cos(phi)``.  Note: the FK returns sagittal-plane
        displacements; to obtain along-profile ux for 2D FD coupling,
        multiply by ``cos(phi)`` (done in ``hybrid/background.py``).
    model_max_depth : float, optional
        Default ``z_init`` (maximum model depth ``Zn``) when ``z_init`` is None.
    freq_damping : float
        Optional imaginary-frequency stabilizer ``eps`` (rad/s). 0 disables it.
    method : str
        Propagation method: ``"kennett_numba"`` (Kennett reflectivity,
        Numba-accelerated; default), ``"kennett"`` (Kennett reflectivity,
        pure NumPy), ``"haskell"`` (Thomson-Haskell propagator matrix,
        NumPy), or ``"haskell_numba"`` (Thomson-Haskell, Numba-accelerated).

    Returns
    -------
    FKResult
    """
    receivers_x = np.atleast_1d(np.asarray(receivers_x, dtype=float))
    nrcv = receivers_x.size
    if receivers_z is None:
        receivers_z = np.zeros(nrcv)
    receivers_z = np.atleast_1d(np.asarray(receivers_z, dtype=float))
    if receivers_z.size != nrcv:
        raise ValueError("receivers_x and receivers_z must have equal length.")

    # Resolve slowness.
    if (p_si is None) == (p_deg is None):
        raise ValueError("Provide exactly one of p_si or p_deg.")
    if p_si is None:
        p_si = rayp_deg_to_si(p_deg)
    if abs(p_si) < 1e-12:
        p_si = 1e-12 * (1.0 if p_si >= 0 else -1.0)

    # Resolve the vertical reference depth (must lie within the half-space).
    if z_init is None:
        if model_max_depth is None:
            raise ValueError("Provide z_init or model_max_depth.")
        z_init = model_max_depth
    if z_init < model.z_top[model.halfspace_index]:
        raise ValueError(
            "z_init must lie within the half-space "
            f"(>= {model.z_top[model.halfspace_index]:.1f} m)."
        )

    # Frequency axis (one-sided, matches rfft/irfft).
    freqs = rfftfreq(nt, d=dt)
    omega = 2.0 * np.pi * freqs
    nfreq = omega.size
    # Avoid division by zero at DC inside the propagator; DC is zeroed later.
    omega_safe = omega.copy()
    omega_safe[omega_safe == 0.0] = 1e-30

    use_damping = freq_damping > 0.0
    omega_c = omega_safe - 1j * freq_damping if use_damping else omega_safe

    amp = incident_amplitude(p_si, source.incidence, model)

    # Select propagator functions based on method.
    method = method.lower()
    if method == "haskell":
        _surface_fn = surface_state
        _depth_fn = state_at_depth
    elif method == "kennett":
        _surface_fn = surface_state_kennett
        _depth_fn = state_at_depth_kennett
    elif method == "kennett_numba":
        from .propagators.kennett_numba import surface_state_kennett_numba
        _surface_fn = surface_state_kennett_numba
        _depth_fn = None  # handled via batch below
    elif method == "haskell_numba":
        from .propagators.haskell_numba import surface_state_numba
        _surface_fn = surface_state_numba
        _depth_fn = None  # handled via batch below
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose 'haskell', 'haskell_numba', 'kennett', or 'kennett_numba'."
        )

    # Frequency-domain displacement response (per receiver depth).
    # Group receivers by depth to reuse the surface/continuation state.
    ux_spec = np.zeros((nrcv, nfreq), dtype=complex)
    uz_spec = np.zeros((nrcv, nfreq), dtype=complex)

    unique_depths = np.unique(receivers_z)
    state_cache = {}

    if method in ("kennett_numba", "haskell_numba") and len(unique_depths) > 1:
        if method == "kennett_numba":
            from .propagators.kennett_numba import states_at_depths_kennett_numba
            batch_fn = states_at_depths_kennett_numba
        else:
            from .propagators.haskell_numba import states_at_depths_numba
            batch_fn = states_at_depths_numba
        res = batch_fn(
            model, omega_c, p_si, source.incidence, z_init,
            unique_depths, amp=amp
        )
        for idx, zr in enumerate(unique_depths):
            state_cache[zr] = (res[:, idx, 0], res[:, idx, 1])
    else:
        for zr in unique_depths:
            if zr == 0.0:
                y = _surface_fn(model, omega_c, p_si, source.incidence,
                                z_init, amp=amp)
            else:
                if method == "kennett_numba":
                    from .propagators.kennett_numba import (
                        states_at_depths_kennett_numba,
                    )
                    y = states_at_depths_kennett_numba(
                        model, omega_c, p_si, source.incidence, z_init,
                        np.array([zr]), amp=amp
                    )[:, 0, :]
                elif method == "haskell_numba":
                    from .propagators.haskell_numba import states_at_depths_numba
                    y = states_at_depths_numba(
                        model, omega_c, p_si, source.incidence, z_init,
                        np.array([zr]), amp=amp
                    )[:, 0, :]
                else:
                    y = _depth_fn(model, omega_c, p_si, source.incidence,
                                  z_init, zr, amp=amp)
            state_cache[zr] = (y[:, 0], y[:, 1])

    # Source spectrum.
    if use_damping:
        src_spec = _analytic_ricker_spectrum(
            omega_c, source.f0, source.t0, source.strength
        )
    else:
        src_spec = source.spectrum(nt, dt)

    # Plane-wave moveout per receiver. The along-profile horizontal slowness is
    # -p*cos(phi): the back-azimuth points from station to source, opposite to
    # the propagation direction, so arrival time increases away from the source.
    p_along = -p_si * np.cos(phi)
    for j in range(nrcv):
        gx, gz = state_cache[receivers_z[j]]
        moveout = np.exp(-1j * omega_c * p_along * (receivers_x[j] - x0_plane))
        ux_spec[j] = src_spec * gx * moveout
        uz_spec[j] = src_spec * gz * moveout

    # Zero the DC (and any non-finite) components.
    ux_spec[:, omega == 0.0] = 0.0
    uz_spec[:, omega == 0.0] = 0.0
    ux_spec[~np.isfinite(ux_spec)] = 0.0
    uz_spec[~np.isfinite(uz_spec)] = 0.0

    ux = irfft(ux_spec, n=nt, axis=1)
    uz = irfft(uz_spec, n=nt, axis=1)

    if use_damping:
        t = np.arange(nt) * dt
        ux *= np.exp(freq_damping * t)[None, :]
        uz *= np.exp(freq_damping * t)[None, :]

    t = np.arange(nt) * dt
    return FKResult(
        t=t, ux=ux, uz=uz,
        receivers_x=receivers_x, receivers_z=receivers_z,
        p_si=p_si, z_init=z_init,
    )
