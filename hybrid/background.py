"""FK background-wavefield computation at coupling-zone boundary nodes.

Pre-computes the 1D plane-wave response u0(x, z, t) at all grid points
in the coupling zone (left, right, bottom boundaries) for all time steps.
These arrays are consumed by the wavefield-injection step during FD
time-marching.

The computation uses :func:`fk.propagators.haskell_numba.states_at_depths_numba`
for efficient parallel evaluation over frequencies and depths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.fft import irfft as _scipy_irfft

    def _parallel_irfft(x, n, axis=-1):
        return _scipy_irfft(x, n=n, axis=axis, workers=-1)
except ImportError:
    def _parallel_irfft(x, n, axis=-1):
        return np.fft.irfft(x, n=n, axis=axis)

from fk.model import FKLayerModel
from fk.source import RickerSource
from fk.geometry import incidence_angle
from fk.propagators.kennett_numba import states_at_depths_kennett_numba as states_at_depths_numba

from .config import HybridConfig


def synthesis_length(nt: int, pad_factor: float) -> int:
    """Choose the time-axis length ``nt_syn`` for the FK boundary-wavefield IFFT.

    Purpose
    -------
    The FK background (boundary) wavefield is built in the frequency domain and
    brought to the time domain by an inverse real FFT. That IFFT length defines
    ``nt_syn``, the number of synthesis samples used before the result is
    truncated back to ``nt`` for the FD loop. This helper maps a user-facing
    ``pad_factor`` to a concrete, FFT-efficient ``nt_syn``.

    Problem it addresses (FFT time-wraparound)
    ------------------------------------------
    An IFFT of length ``N`` produces a *periodic* time series with period
    ``N * dt``. Any boundary arrival whose travel time exceeds ``nt * dt``
    (the record length) aliases ("wraps") back to ``t ~ 0`` and would then be
    injected as spurious early-time energy. Synthesising on a longer axis
    ``nt_syn > nt`` (i.e. ``pad_factor > 1``) pushes the wrap period
    ``nt_syn * dt`` beyond the latest physical arrival + coda; the clean
    ``[0, nt)`` portion is then kept and the rest discarded.

    Why an FFT-friendly length (``next_fast_len``)
    ----------------------------------------------
    rfft/irfft runtime degrades toward O(N^2) for prime or large-prime-factor
    lengths and is fastest for "smooth" numbers (5-smooth, i.e. factors 2/3/5,
    and powers of two). We therefore round the *minimum required* length
    (``pad_factor * nt``) UP to the next fast size via
    :func:`scipy.fft.next_fast_len` (``real=True`` matches the real irfft
    backend in :mod:`fk.fft_backend`), falling back to the next power of two
    if SciPy is unavailable. This keeps the longer synthesis cheap while
    wasting little padding (a 5-smooth size is usually only slightly larger
    than ``pad_factor * nt``, and much smaller than the next power of two).

    Parameters
    ----------
    nt : int
        Output number of time samples (the FD record length).
    pad_factor : float
        Multiplier for the synthesis window. ``<= 1.0`` disables padding and
        returns ``nt`` exactly (no-op; the default, back-compatible behaviour).
        Values ``> 1.0`` (e.g. ``2.0``) request an ~``pad_factor``x longer
        synthesis axis to remove wraparound.

    Returns
    -------
    int
        Synthesis length: ``nt`` when ``pad_factor <= 1.0``; otherwise the
        smallest FFT-efficient size ``>= ceil(pad_factor * nt)``.
    """
    # pad_factor <= 1.0 is the explicit no-op path: synthesise at exactly nt
    # (bit-for-bit the original behaviour, no extra FK cost).
    if pad_factor <= 1.0:
        return nt
    target = int(np.ceil(pad_factor * nt))
    try:
        from scipy.fft import next_fast_len
        # real=True selects the optimal length for a real (irfft) transform.
        return int(next_fast_len(target, real=True))
    except ImportError:
        # Fallback: next power of two (avoids slow prime-length transforms).
        return int(2 ** int(np.ceil(np.log2(target))))


def _models_equal(m1: FKLayerModel, m2: FKLayerModel) -> bool:
    """Check if two FK layer models have identical parameters."""
    return (m1 is m2) or (
        np.array_equal(m1.vp, m2.vp) and
        np.array_equal(m1.vs, m2.vs) and
        np.array_equal(m1.rho, m2.rho) and
        np.array_equal(m1.z_top, m2.z_top)
    )


@dataclass
class BoundaryWavefield:
    """Pre-computed background displacement at coupling-zone nodes.

    Convention: the "left" and "right" boundary arrays cover the full
    depth range of the model (nz rows), at 2*nbd columns each (nbd in
    the regular domain + nbd extending into the PML).  The "bottom" array
    covers all nx columns at 2*nbd depth rows.

    Attributes
    ----------
    ux_left, uz_left : ndarray, shape (nz, 2*nbd, nt), float32
    ux_right, uz_right : ndarray, shape (nz, 2*nbd, nt), float32
    ux_bottom, uz_bottom : ndarray, shape (nx + 2*nbd, 2*nbd, nt), float32
        First axis covers x from ``-nbd`` to ``nx+nbd-1`` (extended into
        lateral PMLs to handle corner regions).
    """

    ux_left: np.ndarray
    uz_left: np.ndarray
    ux_right: np.ndarray
    uz_right: np.ndarray
    ux_bottom: np.ndarray
    uz_bottom: np.ndarray


def compute_boundary_wavefield(
    config: HybridConfig,
    model_left: FKLayerModel,
    model_right: FKLayerModel,
    source: RickerSource,
    p_si: float,
    phi: float = 0.0,
    z_init: float | None = None,
    freq_damping: float = 0.0,
    nt_syn: int | None = None,
) -> BoundaryWavefield:
    """Compute FK background wavefield at all coupling-zone grid points.

    Parameters
    ----------
    config : HybridConfig
        Grid and simulation parameters.
    model_left, model_right : FKLayerModel
        1D layered models for left and right boundaries. The half-space
        must be identical.
    source : RickerSource
        Source wavelet and incidence type.
    p_si : float
        Horizontal slowness (s/m).
    phi : float
        Angle between the back-azimuth and the profile azimuth (rad),
        i.e. ``phi = baz - az_org``. Horizontal moveout uses
        ``p_along = -p * cos(phi)`` and the BWF ux component is
        projected from the sagittal plane onto the profile by
        ``cos(phi)``.
    z_init : float, optional
        Plane-wave initialization depth (m). Defaults to a depth
        that ensures the wavefront arrives at the bottom boundary.
    freq_damping : float
        Imaginary-frequency stabilizer (rad/s). 0 disables.
    nt_syn : int, optional
        Anti-wraparound synthesis length for the IFFT. The frequency axis,
        source spectrum and FK states are evaluated on ``nt_syn`` samples and
        the time series is then TRUNCATED back to ``nt`` before being returned,
        so the downstream FD loop sees exactly ``nt`` samples (its cost/length
        is unchanged). ``None`` (or a value ``<= nt``) means use ``nt`` and
        perform no padding. A larger ``nt_syn`` lengthens the IFFT period
        (``nt_syn * dt``) so late boundary arrivals no longer alias into the
        early-time window. See :func:`synthesis_length` for how this value is
        chosen (FFT-friendly) from a ``pad_factor``.

    Returns
    -------
    BoundaryWavefield
    """
    nbd = config.nbd
    nx, nz = config.nx, config.nz
    dt, nt = config.dt, config.nt
    n_syn = nt if nt_syn is None else int(nt_syn)
    dx, dz = config.dx, config.dz

    half = model_left.halfspace_index
    v_half = (model_left.vp[half] if source.incidence == "P"
              else model_left.vs[half])

    if z_init is None:
        theta = incidence_angle(p_si, v_half)
        profile_length = (nx - 1) * dx
        z_init = config.zn + 0.5 * profile_length * np.tan(theta)

    # Frequency axis (one-sided FFT) on the synthesis grid (n_syn >= nt)
    freqs = np.fft.rfftfreq(n_syn, d=dt)
    omega = 2.0 * np.pi * freqs
    nfreq = omega.size
    omega_safe = omega.copy()
    omega_safe[omega_safe == 0.0] = 1e-30

    use_damping = freq_damping > 0.0
    omega_c = omega_safe - 1j * freq_damping if use_damping else omega_safe

    # Source spectrum (evaluated on the synthesis grid)
    if use_damping:
        omega_p = 2.0 * np.pi * source.f0
        src_spec = (
            source.strength
            * (2.0 / np.sqrt(np.pi))
            * (omega_c ** 2 / omega_p ** 3)
            * np.exp(-((omega_c / omega_p) ** 2))
            * np.exp(-1j * omega_c * source.t0)
        )
    else:
        src_spec = source.spectrum(n_syn, dt)

    # Along-profile slowness component
    p_along = -p_si * np.cos(phi)

    # Moveout reference x = profile center. This pairs with the plane-wave
    # init datum z_o = zn + (L/2)*tan(theta): the time origin t=0 is when the
    # wavefront passes (x_center, zn), removing the spurious (L/2)*p offset.
    # Must match the x0_plane convention used by the FK engine reference.
    x0_plane = 0.5 * (config.x0 + config.xn)

    # Incident amplitude
    amp = float(np.sin(incidence_angle(p_si, v_half)))

    # --- Depth arrays for lateral boundaries (all nz depths) ---
    z_all = config.z_coords  # (nz,)

    # Bottom boundary depths: nbd rows at bottom of regular + nbd into PML
    z_bottom = config.z0 + np.arange(nz - nbd, nz + nbd) * dz  # (2*nbd,)

    # --- X-coordinates for lateral boundaries ---
    # Left: nbd columns in PML (ix = -nbd..-1) + nbd in regular (ix = 0..nbd-1)
    x_left = config.x0 + np.arange(-nbd, nbd) * dx   # (2*nbd,)
    # Right: nbd in regular (ix = nx-nbd..nx-1) + nbd in PML (ix = nx..nx+nbd-1)
    x_right = config.x0 + np.arange(nx - nbd, nx + nbd) * dx  # (2*nbd,)

    # --- Compute FK states at all required depths ---
    omega_real = np.real(omega_safe).astype(np.float64)

    same_lr = _models_equal(model_left, model_right)

    states_left_full = states_at_depths_numba(
        model_left, omega_real, p_si, source.incidence, z_init, z_all, amp=amp
    )  # (nfreq, nz, 4)

    states_right_full = (
        states_left_full if same_lr
        else states_at_depths_numba(
            model_right, omega_real, p_si, source.incidence, z_init, z_all, amp=amp
        )
    )  # (nfreq, nz, 4)

    states_bottom_left = states_at_depths_numba(
        model_left, omega_real, p_si, source.incidence, z_init, z_bottom, amp=amp
    )  # (nfreq, 2*nbd, 4)

    states_bottom_right = (
        states_bottom_left if same_lr
        else states_at_depths_numba(
            model_right, omega_real, p_si, source.incidence, z_init, z_bottom, amp=amp
        )
    )  # (nfreq, 2*nbd, 4)

    # --- Assemble boundary wavefields with moveout and IFFT ---
    ux_left, uz_left = _lateral_boundary_to_time(
        states_left_full, src_spec, omega_c, omega, p_along, x0_plane,
        x_left, nt, n_syn, dt, freq_damping, use_damping
    )  # (nz, 2*nbd, nt)

    ux_right, uz_right = _lateral_boundary_to_time(
        states_right_full, src_spec, omega_c, omega, p_along, x0_plane,
        x_right, nt, n_syn, dt, freq_damping, use_damping
    )  # (nz, 2*nbd, nt)

    ux_bottom, uz_bottom = _bottom_boundary_to_time(
        states_bottom_left, states_bottom_right,
        src_spec, omega_c, omega, p_along, x0_plane,
        config, nt, n_syn, dt, freq_damping, use_damping
    )  # (nx, 2*nbd, nt)

    # Project sagittal-plane ux onto the 2D profile direction.
    # The FK propagator returns displacements in the sagittal plane (source-
    # receiver vertical plane), but the 2D FD operates along the profile.
    # The along-profile component is ux_profile = ux_sagittal * cos(phi).
    cos_phi = np.float32(np.cos(phi))
    return BoundaryWavefield(
        ux_left=(ux_left * cos_phi).astype(np.float32),
        uz_left=uz_left.astype(np.float32),
        ux_right=(ux_right * cos_phi).astype(np.float32),
        uz_right=uz_right.astype(np.float32),
        ux_bottom=(ux_bottom * cos_phi).astype(np.float32),
        uz_bottom=uz_bottom.astype(np.float32),
    )


def _lateral_boundary_to_time(states, src_spec, omega_c, omega, p_along,
                              x0_plane, x_coords, nt, nt_syn, dt, freq_damping,
                              use_damping):
    """Convert FK states at multiple depths to time-domain for a lateral boundary.

    Parameters
    ----------
    states : ndarray, shape (nfreq, n_depths, 4)
    x_coords : ndarray, shape (n_cols,)
    nt : int
        Output number of samples (after truncation).
    nt_syn : int
        IFFT synthesis length (>= nt) for anti-wraparound zero-padding.

    Returns
    -------
    ux, uz : ndarray, shape (n_depths, n_cols, nt)
    """
    nfreq, n_depths, _ = states.shape
    n_cols = x_coords.size

    # Vectorized over all (depth, column) pairs
    # moveout: (n_cols, nfreq)
    moveout = np.exp(
        -1j * omega_c[np.newaxis, :] * p_along
        * (x_coords[:, np.newaxis] - x0_plane)
    )  # (n_cols, nfreq)

    # State ux/uz: (nfreq, n_depths) -> transpose to (n_depths, nfreq)
    gx = states[:, :, 0].T  # (n_depths, nfreq)
    gz = states[:, :, 1].T  # (n_depths, nfreq)

    # Outer product: (n_depths, n_cols, nfreq)
    ux_spec = (src_spec[np.newaxis, np.newaxis, :]
               * gx[:, np.newaxis, :]
               * moveout[np.newaxis, :, :])
    uz_spec = (src_spec[np.newaxis, np.newaxis, :]
               * gz[:, np.newaxis, :]
               * moveout[np.newaxis, :, :])

    # Zero DC and sanitize non-finite values
    dc_mask = (omega == 0.0)
    ux_spec[:, :, dc_mask] = 0.0
    uz_spec[:, :, dc_mask] = 0.0
    np.nan_to_num(ux_spec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(uz_spec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    ux_lat = _parallel_irfft(ux_spec, n=nt_syn, axis=-1)
    uz_lat = _parallel_irfft(uz_spec, n=nt_syn, axis=-1)

    if use_damping:
        t_arr = np.arange(nt_syn) * dt
        taper = np.exp(freq_damping * t_arr)
        ux_lat *= taper[np.newaxis, np.newaxis, :]
        uz_lat *= taper[np.newaxis, np.newaxis, :]

    # Truncate the (possibly zero-padded) synthesis window back to nt
    return ux_lat[:, :, :nt], uz_lat[:, :, :nt]


def _bottom_boundary_to_time(states_left, states_right, src_spec, omega_c,
                             omega, p_along, x0_plane, config, nt, nt_syn, dt,
                             freq_damping, use_damping):
    """Compute bottom-boundary wavefield using left/right models.

    Left model for the left half of the profile, right model for the right half.
    Extends into left/right PML by nbd columns on each side to cover corner
    PML regions (matching Fortran's uxb_pml dimension 1-nbd:nx+nbd).

    Returns
    -------
    ux, uz : ndarray, shape (nx + 2*nbd, 2*nbd, nt)
        First axis covers x-indices from -nbd to nx+nbd-1 (extended grid).
    """
    nx = config.nx
    nbd = config.nbd
    nfreq = omega_c.size
    n_depth_rows = 2 * nbd

    # Extended x-coordinates: include nbd PML columns on each side
    nx_ext = nx + 2 * nbd
    x_coords = config.x0 + np.arange(-nbd, nx + nbd) * config.dx  # (nx_ext,)
    x_mid = 0.5 * (config.x0 + config.xn)

    # Moveout for all x-coordinates: (nx_ext, nfreq)
    moveout = np.exp(
        -1j * omega_c[np.newaxis, :] * p_along
        * (x_coords[:, np.newaxis] - x0_plane)
    )

    # Green's functions at each depth row: (nfreq, n_depth_rows, 2)
    # Split by left/right half
    left_mask = x_coords <= x_mid

    # states shape: (nfreq, n_depth_rows, 4) -> extract gx, gz
    gx_left = states_left[:, :, 0]   # (nfreq, n_depth_rows)
    gz_left = states_left[:, :, 1]
    gx_right = states_right[:, :, 0]
    gz_right = states_right[:, :, 1]

    ux_spec = np.zeros((nx_ext, n_depth_rows, nfreq), dtype=complex)
    uz_spec = np.zeros((nx_ext, n_depth_rows, nfreq), dtype=complex)

    # Vectorized: left half
    n_left = np.sum(left_mask)
    if n_left > 0:
        # (n_left, 1, nfreq) * (1, n_depth_rows, nfreq)
        ux_spec[left_mask] = (
            src_spec[np.newaxis, np.newaxis, :]
            * gx_left.T[np.newaxis, :, :]   # (1, n_depth_rows, nfreq)
            * moveout[left_mask, np.newaxis, :]  # (n_left, 1, nfreq)
        )
        uz_spec[left_mask] = (
            src_spec[np.newaxis, np.newaxis, :]
            * gz_left.T[np.newaxis, :, :]
            * moveout[left_mask, np.newaxis, :]
        )

    # Right half
    right_mask = ~left_mask
    n_right = np.sum(right_mask)
    if n_right > 0:
        ux_spec[right_mask] = (
            src_spec[np.newaxis, np.newaxis, :]
            * gx_right.T[np.newaxis, :, :]
            * moveout[right_mask, np.newaxis, :]
        )
        uz_spec[right_mask] = (
            src_spec[np.newaxis, np.newaxis, :]
            * gz_right.T[np.newaxis, :, :]
            * moveout[right_mask, np.newaxis, :]
        )

    # Zero DC and sanitize non-finite values
    dc_mask = (omega == 0.0)
    ux_spec[:, :, dc_mask] = 0.0
    uz_spec[:, :, dc_mask] = 0.0
    np.nan_to_num(ux_spec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(uz_spec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    ux_bot = _parallel_irfft(ux_spec, n=nt_syn, axis=-1)
    uz_bot = _parallel_irfft(uz_spec, n=nt_syn, axis=-1)

    if use_damping:
        t_arr = np.arange(nt_syn) * dt
        taper = np.exp(freq_damping * t_arr)
        ux_bot *= taper[np.newaxis, np.newaxis, :]
        uz_bot *= taper[np.newaxis, np.newaxis, :]

    # Truncate the (possibly zero-padded) synthesis window back to nt
    return ux_bot[:, :, :nt], uz_bot[:, :, :nt]


# ===================================================================
# Source-independent FK Green's function caching
# ===================================================================

@dataclass
class FKGreensCache:
    """Source-independent FK Green's functions at coupling-zone boundaries.

    Stores propagator-matrix states that depend only on the velocity model,
    ray parameter, and incidence type — NOT on the source wavelet.  Reuse
    this cache when sweeping over different source wavelets / spectra to
    avoid redundant FK propagator-matrix computations.
    """
    states_left_full: np.ndarray
    states_right_full: np.ndarray
    states_bottom_left: np.ndarray
    states_bottom_right: np.ndarray
    omega: np.ndarray
    omega_safe: np.ndarray
    p_along: float
    x0_plane: float
    z_init: float
    x_left: np.ndarray
    x_right: np.ndarray
    amp: float
    cos_phi: float = 1.0
    # Anti-wraparound IFFT synthesis length baked into this cache. The cached
    # FK states are evaluated on the ``nt_syn``-sample frequency axis;
    # :func:`boundary_wavefield_from_greens` performs its IFFT at this length
    # and then truncates back to ``config.nt``. 0 means "use config.nt"
    # (no padding), preserving the original behaviour.
    nt_syn: int = 0


def compute_fk_greens(
    config: HybridConfig,
    model_left: FKLayerModel,
    model_right: FKLayerModel,
    p_si: float,
    incidence: str,
    *,
    phi: float = 0.0,
    z_init: float | None = None,
    nt_syn: int | None = None,
) -> FKGreensCache:
    """Pre-compute source-independent FK Green's functions at boundary nodes.

    This performs the expensive propagator-matrix evaluation over all
    frequencies and depths.  The result can be reused with
    :func:`boundary_wavefield_from_greens` for different source wavelets
    without repeating the FK propagation.

    Parameters
    ----------
    phi : float
        Angle between the back-azimuth and the profile azimuth (rad),
        i.e. ``phi = baz - az_org``.  Stored as ``cos_phi`` in the
        returned cache for sagittal-to-profile projection of BWF ux.
    nt_syn : int, optional
        Anti-wraparound synthesis length (see :func:`compute_boundary_wavefield`
        and :func:`synthesis_length`). The FK states are evaluated on this many
        frequency samples and the value is stored in the returned cache so that
        :func:`boundary_wavefield_from_greens` performs its IFFT at ``nt_syn``
        and truncates back to ``config.nt``. ``None`` (or ``<= nt``) means use
        ``nt`` (no padding).
    """
    nbd = config.nbd
    nx, nz = config.nx, config.nz
    dt, nt = config.dt, config.nt
    n_syn = nt if nt_syn is None else int(nt_syn)
    dx = config.dx

    half = model_left.halfspace_index
    v_half = (model_left.vp[half] if incidence.upper() == "P"
              else model_left.vs[half])

    if z_init is None:
        theta = incidence_angle(p_si, v_half)
        profile_length = (nx - 1) * dx
        z_init = config.zn + 0.5 * profile_length * np.tan(theta)

    freqs = np.fft.rfftfreq(n_syn, d=dt)
    omega = 2.0 * np.pi * freqs
    omega_safe = omega.copy()
    omega_safe[omega_safe == 0.0] = 1e-30

    p_along = -p_si * np.cos(phi)
    # Moveout reference x = profile center (see compute_boundary_wavefield):
    # pairs with the z_o = zn + (L/2)*tan(theta) datum so t=0 is when the
    # wavefront passes (x_center, zn). Matches the FK engine x0_plane.
    x0_plane = 0.5 * (config.x0 + config.xn)
    amp = float(np.sin(incidence_angle(p_si, v_half)))

    z_all = config.z_coords
    z_bottom = config.z0 + np.arange(nz - nbd, nz + nbd) * config.dz
    x_left = config.x0 + np.arange(-nbd, nbd) * dx
    x_right = config.x0 + np.arange(nx - nbd, nx + nbd) * dx

    omega_real = np.real(omega_safe).astype(np.float64)
    same_lr = _models_equal(model_left, model_right)

    states_left_full = states_at_depths_numba(
        model_left, omega_real, p_si, incidence, z_init, z_all, amp=amp
    )
    states_right_full = (
        states_left_full if same_lr
        else states_at_depths_numba(
            model_right, omega_real, p_si, incidence, z_init, z_all, amp=amp
        )
    )
    states_bottom_left = states_at_depths_numba(
        model_left, omega_real, p_si, incidence, z_init, z_bottom, amp=amp
    )
    states_bottom_right = (
        states_bottom_left if same_lr
        else states_at_depths_numba(
            model_right, omega_real, p_si, incidence, z_init, z_bottom, amp=amp
        )
    )

    return FKGreensCache(
        states_left_full=states_left_full,
        states_right_full=states_right_full,
        states_bottom_left=states_bottom_left,
        states_bottom_right=states_bottom_right,
        omega=omega,
        omega_safe=omega_safe,
        p_along=float(p_along),
        x0_plane=float(x0_plane),
        z_init=float(z_init),
        x_left=x_left,
        x_right=x_right,
        amp=float(amp),
        cos_phi=float(np.cos(phi)),
        nt_syn=int(n_syn),
    )


def boundary_wavefield_from_greens(
    config: HybridConfig,
    greens: FKGreensCache,
    source: RickerSource,
    *,
    freq_damping: float = 0.0,
) -> BoundaryWavefield:
    """Build time-domain boundary wavefield from cached FK Green's functions.

    Only the source-spectrum multiplication and IFFT are performed — the
    expensive propagator-matrix computation is skipped entirely.

    The anti-wraparound synthesis length is taken from ``greens.nt_syn`` (set
    when the cache was built). The IFFT runs at ``nt_syn`` samples and the
    result is truncated back to ``config.nt``, so the FD loop is unaffected.
    """
    nt = config.nt
    dt = config.dt
    # Synthesis length stored in the cache (anti-wraparound zero-pad);
    # 0/absent means "use nt" (no padding).
    n_syn = greens.nt_syn if greens.nt_syn else nt
    omega = greens.omega
    omega_safe = greens.omega_safe

    use_damping = freq_damping > 0.0
    omega_c = omega_safe - 1j * freq_damping if use_damping else omega_safe

    if use_damping:
        omega_p = 2.0 * np.pi * source.f0
        src_spec = (
            source.strength
            * (2.0 / np.sqrt(np.pi))
            * (omega_c ** 2 / omega_p ** 3)
            * np.exp(-((omega_c / omega_p) ** 2))
            * np.exp(-1j * omega_c * source.t0)
        )
    else:
        src_spec = source.spectrum(n_syn, dt)

    ux_left, uz_left = _lateral_boundary_to_time(
        greens.states_left_full, src_spec, omega_c, omega,
        greens.p_along, greens.x0_plane, greens.x_left,
        nt, n_syn, dt, freq_damping, use_damping,
    )
    ux_right, uz_right = _lateral_boundary_to_time(
        greens.states_right_full, src_spec, omega_c, omega,
        greens.p_along, greens.x0_plane, greens.x_right,
        nt, n_syn, dt, freq_damping, use_damping,
    )
    ux_bottom, uz_bottom = _bottom_boundary_to_time(
        greens.states_bottom_left, greens.states_bottom_right,
        src_spec, omega_c, omega, greens.p_along, greens.x0_plane,
        config, nt, n_syn, dt, freq_damping, use_damping,
    )

    cos_phi = np.float32(greens.cos_phi)
    return BoundaryWavefield(
        ux_left=(ux_left * cos_phi).astype(np.float32),
        uz_left=uz_left.astype(np.float32),
        ux_right=(ux_right * cos_phi).astype(np.float32),
        uz_right=uz_right.astype(np.float32),
        ux_bottom=(ux_bottom * cos_phi).astype(np.float32),
        uz_bottom=uz_bottom.astype(np.float32),
    )


def compute_regular_wavefield(
    config: HybridConfig,
    model: FKLayerModel,
    source: RickerSource,
    p_si: float,
    phi: float = 0.0,
    z_init: float | None = None,
    freq_damping: float = 0.0,
    nt_syn: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """FK background u₀ on all regular-domain nodes (diagnostic / validation).

    Returns ``(ux, uz)``, each shape ``(nz, nx, nt)`` in float32.
    """
    nx, nz = config.nx, config.nz
    dt, nt = config.dt, config.nt
    n_syn = nt if nt_syn is None else int(nt_syn)
    dx = config.dx

    half = model.halfspace_index
    v_half = (model.vp[half] if source.incidence == "P"
              else model.vs[half])

    if z_init is None:
        theta = incidence_angle(p_si, v_half)
        profile_length = (nx - 1) * dx
        z_init = config.zn + 0.5 * profile_length * np.tan(theta)

    freqs = np.fft.rfftfreq(n_syn, d=dt)
    omega = 2.0 * np.pi * freqs
    omega_safe = omega.copy()
    omega_safe[omega_safe == 0.0] = 1e-30

    use_damping = freq_damping > 0.0
    omega_c = omega_safe - 1j * freq_damping if use_damping else omega_safe

    if use_damping:
        omega_p = 2.0 * np.pi * source.f0
        src_spec = (
            source.strength
            * (2.0 / np.sqrt(np.pi))
            * (omega_c ** 2 / omega_p ** 3)
            * np.exp(-((omega_c / omega_p) ** 2))
            * np.exp(-1j * omega_c * source.t0)
        )
    else:
        src_spec = source.spectrum(n_syn, dt)

    p_along = -p_si * np.cos(phi)
    x0_plane = 0.5 * (config.x0 + config.xn)
    amp = float(np.sin(incidence_angle(p_si, v_half)))

    z_all = config.z_coords
    x_all = config.x_coords
    omega_real = np.real(omega_safe).astype(np.float64)

    states = states_at_depths_numba(
        model, omega_real, p_si, source.incidence, z_init, z_all, amp=amp
    )

    ux_reg, uz_reg = _lateral_boundary_to_time(
        states, src_spec, omega_c, omega, p_along, x0_plane,
        x_all, nt, n_syn, dt, freq_damping, use_damping,
    )

    cos_phi = np.float32(np.cos(phi))
    return (ux_reg * cos_phi).astype(np.float32), uz_reg.astype(np.float32)
