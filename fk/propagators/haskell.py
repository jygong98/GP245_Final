"""Propagator-matrix (Haskell/Thomson) core of the FK method.

Implements the P-SV plane-wave response of a 1D layered model following
Liu et al. (2025, SRL) Appendix A. The state vector is

    y = [ux_FK, uz_FK, sigma_hz, sigma_zz]

and per layer the eigenvector matrix ``R`` (eq. A5/A7), phase matrix ``H``
(eq. A8) and layer propagator ``P = R H R^-1`` (eq. A6) are assembled. All
matrices are batched over angular frequency with shape ``(nfreq, 4, 4)``.

Conventions: positive z is downward; a teleseismic wave arrives from below as
an up-going wave in the half-space. Coefficient ordering is ``[DS, US, DP, UP]``
(down-going S, up-going S, down-going P, up-going P).
"""

from __future__ import annotations

import numpy as np

from ..geometry import vertical_wavenumbers, incidence_angle
from ..model import FKLayerModel

# Coefficient column indices in [DS, US, DP, UP].
IDX_DS, IDX_US, IDX_DP, IDX_UP = 0, 1, 2, 3


def eigen_matrix_R(omega, p_si, vp, vs, rho):
    """Eigenvector matrix ``R`` for one layer, shape ``(nfreq, 4, 4)``.

    Columns correspond to [DS, US, DP, UP]; see eq. A5/A7.
    """
    kh, kp_z, ks_z, gamma = vertical_wavenumbers(omega, p_si, vp, vs)
    mu = rho * vs ** 2
    nf = kh.shape[0]
    R = np.zeros((nf, 4, 4), dtype=complex)

    ks_kh = ks_z / kh
    kp_kh = kp_z / kh
    img = 1j

    # Row 0: ux
    R[:, 0, IDX_DS] = ks_kh
    R[:, 0, IDX_US] = -ks_kh
    R[:, 0, IDX_DP] = 1.0
    R[:, 0, IDX_UP] = 1.0
    # Row 1: uz
    R[:, 1, IDX_DS] = 1.0
    R[:, 1, IDX_US] = 1.0
    R[:, 1, IDX_DP] = -kp_kh
    R[:, 1, IDX_UP] = kp_kh
    # Row 2: sigma_hz
    R[:, 2, IDX_DS] = -img * mu * gamma / kh
    R[:, 2, IDX_US] = -img * mu * gamma / kh
    R[:, 2, IDX_DP] = -2.0 * img * mu * kp_z
    R[:, 2, IDX_UP] = 2.0 * img * mu * kp_z
    # Row 3: sigma_zz
    R[:, 3, IDX_DS] = -2.0 * img * mu * ks_z
    R[:, 3, IDX_US] = 2.0 * img * mu * ks_z
    R[:, 3, IDX_DP] = img * mu * gamma / kh
    R[:, 3, IDX_UP] = img * mu * gamma / kh
    return R


def phase_matrix_H(omega, p_si, vp, vs, dz):
    """Diagonal phase matrix ``H(dz)``, shape ``(nfreq, 4, 4)`` (eq. A8).

    Diagonal = ``[exp(-i ks_z dz), exp(+i ks_z dz),
                  exp(-i kp_z dz), exp(+i kp_z dz)]``.
    """
    kh, kp_z, ks_z, gamma = vertical_wavenumbers(omega, p_si, vp, vs)
    nf = kh.shape[0]
    H = np.zeros((nf, 4, 4), dtype=complex)
    H[:, IDX_DS, IDX_DS] = np.exp(-1j * ks_z * dz)
    H[:, IDX_US, IDX_US] = np.exp(1j * ks_z * dz)
    H[:, IDX_DP, IDX_DP] = np.exp(-1j * kp_z * dz)
    H[:, IDX_UP, IDX_UP] = np.exp(1j * kp_z * dz)
    return H


def layer_propagator_P(omega, p_si, vp, vs, rho, thickness):
    """Layer propagator ``P = R H(-h) R^-1`` with ``y_{i-1} = P y_i`` (eq. A6).

    ``thickness`` is the layer thickness ``h``; the phase offset is
    ``z_{i-1} - z_i = -h``.
    """
    R = eigen_matrix_R(omega, p_si, vp, vs, rho)
    H = phase_matrix_H(omega, p_si, vp, vs, -thickness)
    Rinv = np.linalg.inv(R)
    return R @ H @ Rinv


def build_propagator(model: FKLayerModel, omega, p_si, z_init):
    """Assemble the full propagator ``A`` from the half-space to the surface.

    ``A = P_0 P_1 ... P_{n-1} R_half H(z_top_half - z_init)`` (eq. A9/A10),
    where finite layers are entries ``0..n-1`` and the half-space is the last
    entry. ``y(z0) = A @ [DS, US, DP, UP]`` with coefficients defined at the
    plane-wave initialization depth ``z_init`` in the half-space.

    Returns
    -------
    numpy.ndarray
        Propagator ``A`` of shape ``(nfreq, 4, 4)``.
    """
    omega = np.asarray(omega, dtype=float)
    nf = omega.shape[0]
    A = np.broadcast_to(np.eye(4, dtype=complex), (nf, 4, 4)).copy()

    th = model.thickness
    for i in range(model.n_finite_layers):
        P = layer_propagator_P(
            omega, p_si, model.vp[i], model.vs[i], model.rho[i], th[i]
        )
        A = A @ P

    half = model.halfspace_index
    R_half = eigen_matrix_R(
        omega, p_si, model.vp[half], model.vs[half], model.rho[half]
    )
    dz_half = model.z_top[half] - z_init
    H_half = phase_matrix_H(
        omega, p_si, model.vp[half], model.vs[half], dz_half
    )
    A = A @ R_half @ H_half
    return A


def solve_halfspace_coeffs(A, incidence, amp):
    """Solve free-surface BC for half-space coefficients ``[DS, US, DP, UP]``.

    Traction-free surface => rows 2,3 of ``A @ c`` vanish. For P incidence
    ``US = 0`` and ``UP = amp`` is given (eq. A11/A12); for S incidence
    ``UP = 0`` and ``US = amp`` (eq. A14/A15). The two reflected coefficients
    ``DS, DP`` are solved per frequency.

    Parameters
    ----------
    A : numpy.ndarray
        Propagator, shape ``(nfreq, 4, 4)``.
    incidence : str
        ``"P"`` or ``"S"``.
    amp : float or numpy.ndarray
        Incident-wave amplitude (the free, given coefficient).

    Returns
    -------
    numpy.ndarray
        Coefficient vectors ``c`` of shape ``(nfreq, 4)``.
    """
    nf = A.shape[0]
    incidence = incidence.upper()
    if incidence == "P":
        inc_col = IDX_UP
    elif incidence == "S":
        inc_col = IDX_US
    else:
        raise ValueError("incidence must be 'P' or 'S'.")

    # 2x2 system on (DS, DP) from traction rows (2, 3).
    M = np.empty((nf, 2, 2), dtype=complex)
    M[:, 0, 0] = A[:, 2, IDX_DS]
    M[:, 0, 1] = A[:, 2, IDX_DP]
    M[:, 1, 0] = A[:, 3, IDX_DS]
    M[:, 1, 1] = A[:, 3, IDX_DP]

    rhs = np.empty((nf, 2), dtype=complex)
    rhs[:, 0] = -A[:, 2, inc_col] * amp
    rhs[:, 1] = -A[:, 3, inc_col] * amp

    sol = np.linalg.solve(M, rhs[..., None])[..., 0]  # (nf, 2) -> (DS, DP)

    c = np.zeros((nf, 4), dtype=complex)
    c[:, IDX_DS] = sol[:, 0]
    c[:, IDX_DP] = sol[:, 1]
    c[:, inc_col] = amp
    return c


def incident_amplitude(p_si, incidence, model: FKLayerModel) -> float:
    """Incident coefficient giving unit free-surface displacement of the
    incident wave (paper eqs. A13/A16): ``amp = sin(theta)``.

    ``theta`` is the incidence angle in the half-space, using the P (S)
    velocity for P (S) incidence.
    """
    half = model.halfspace_index
    v = model.vp[half] if incidence.upper() == "P" else model.vs[half]
    return float(np.sin(incidence_angle(p_si, v)))


def surface_state(model: FKLayerModel, omega, p_si, incidence, z_init, amp=None):
    """Free-surface state vector ``y(z0)`` over frequency.

    Returns
    -------
    y0 : numpy.ndarray
        Shape ``(nfreq, 4)``; ``y0[:, 0]`` = ux, ``y0[:, 1]`` = uz.
    """
    if amp is None:
        amp = incident_amplitude(p_si, incidence, model)
    A = build_propagator(model, omega, p_si, z_init)
    c = solve_halfspace_coeffs(A, incidence, amp)
    y0 = np.einsum("fij,fj->fi", A, c)
    return y0


def _layer_index_of_depth(model: FKLayerModel, z: float) -> int:
    """Entry index containing depth ``z`` (half-space for z >= last z_top)."""
    if z <= model.z_top[0]:
        return 0
    idx = int(np.searchsorted(model.z_top, z, side="right") - 1)
    return min(idx, model.halfspace_index)


def state_at_depth(model: FKLayerModel, omega, p_si, incidence, z_init,
                   z_r, amp=None):
    """State vector ``y(z_r)`` at an arbitrary depth via downward continuation.

    Implements eq. A17:
    ``y(z_r) = R_i H(z_r - z_{i-1}) R_i^-1 P_{i-1}^-1 ... P_0^-1 y(z0)``,
    where layer ``i`` (entry index) contains ``z_r`` and ``z_{i-1}`` is its top.

    Returns array of shape ``(nfreq, 4)``.
    """
    omega = np.asarray(omega, dtype=float)
    y0 = surface_state(model, omega, p_si, incidence, z_init, amp=amp)

    i = _layer_index_of_depth(model, z_r)
    th = model.thickness
    # Apply inverse propagators of the finite layers above layer i.
    y = y0.copy()
    for k in range(i):
        P = layer_propagator_P(
            omega, p_si, model.vp[k], model.vs[k], model.rho[k], th[k]
        )
        y = np.einsum("fij,fj->fi", np.linalg.inv(P), y)

    z_top_i = model.z_top[i]
    R = eigen_matrix_R(omega, p_si, model.vp[i], model.vs[i], model.rho[i])
    H = phase_matrix_H(omega, p_si, model.vp[i], model.vs[i], z_r - z_top_i)
    Rinv = np.linalg.inv(R)
    M = R @ H @ Rinv
    y = np.einsum("fij,fj->fi", M, y)
    return y
