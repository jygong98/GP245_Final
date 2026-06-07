"""Kennett's recursive reflectivity method for P-SV plane-wave response.

Computes the full 4-component state vector [ux, uz, sigma_hz, sigma_zz]
at the free surface or arbitrary depths, using the numerically stable
recursive scheme from Kennett (1983, "Seismic Wave Propagation in
Stratified Media"). Unlike the Thomson-Haskell propagator matrix method,
Kennett's approach avoids exponential overflow by working with bounded
reflection/transmission (R/T) coefficients and applying the addition rule
recursively.

The state vector convention matches :mod:`fk.propagators.haskell`:
    y = [ux_FK, uz_FK, sigma_hz, sigma_zz]

Interface R/T coefficients are derived from displacement-stress continuity
using the eigenvector matrices from :mod:`fk.propagators.haskell` (equivalent to solving
the Zoeppritz equations for the P-SV system).

Conventions:
    - Positive z is downward.
    - Incident teleseismic wave from the half-space (up-going).
    - 2x2 sub-system uses [S, P] ordering: index 0 = S wave, index 1 = P wave.
    - Phase delay through layer j: E_j = diag(exp(-i*ksz_j*h_j), exp(-i*kpz_j*h_j)).
"""

from __future__ import annotations

import numpy as np

from ..geometry import vertical_wavenumbers, incidence_angle
from ..model import FKLayerModel

# Column indices matching haskell.py.
IDX_DS, IDX_US, IDX_DP, IDX_UP = 0, 1, 2, 3

# Down-going column indices and up-going column indices in the eigenvector matrix.
_D_COLS = np.array([IDX_DS, IDX_DP])  # [S, P] ordering
_U_COLS = np.array([IDX_US, IDX_UP])  # [S, P] ordering


def _eigen_R_at_interface(omega, p_si, vp, vs, rho):
    """Eigenvector matrix R and vertical wavenumbers for one layer.

    Returns
    -------
    R : ndarray, shape (nfreq, 4, 4)
    kpz, ksz : ndarray, shape (nfreq,)
    """
    from .haskell import eigen_matrix_R
    R = eigen_matrix_R(omega, p_si, vp, vs, rho)
    kh, kpz, ksz, gamma = vertical_wavenumbers(omega, p_si, vp, vs)
    return R, kpz, ksz


def _free_surface_reflection(R0):
    """Free-surface reflection matrix R_F (2x2) from eigenvector matrix of layer 0.

    At z=0, traction-free BC gives sigma_hz = sigma_zz = 0.
    From R_0 @ c = y with y[2:4] = 0:
        R_0_stress_D @ D + R_0_stress_U @ U = 0
        => D = R_F @ U,  R_F = -R_0_stress_D^{-1} @ R_0_stress_U

    Parameters
    ----------
    R0 : ndarray, shape (nfreq, 4, 4)
        Eigenvector matrix for the surface layer.

    Returns
    -------
    R_F : ndarray, shape (nfreq, 2, 2)
        Maps U=[US, UP] → D=[DS, DP] at the free surface.
    """
    R_stress_D = R0[:, 2:4][:, :, _D_COLS]  # (nfreq, 2, 2)
    R_stress_U = R0[:, 2:4][:, :, _U_COLS]  # (nfreq, 2, 2)
    return -np.linalg.solve(R_stress_D, R_stress_U)


def _interface_rt(R_above, R_below):
    """Compute 2x2 R/T matrices at an interface.

    From displacement-stress continuity:
        R_above @ c_above = R_below @ c_below

    For up-going from below (U_below given, D_above = 0):
        [R_above_U | -R_below_D] @ [U_above; D_below] = R_below_U @ U_below

    For down-going from above (D_above given, U_below = 0):
        [R_below_D | -R_above_U] @ [D_below; U_above] = R_above_D @ D_above

    Parameters
    ----------
    R_above, R_below : ndarray, shape (nfreq, 4, 4)

    Returns
    -------
    R_U, R_D, T_U, T_D : ndarray, each shape (nfreq, 2, 2)
        R_U: up-going reflection (U_below → D_below)
        R_D: down-going reflection (D_above → U_above)
        T_U: up-going transmission (U_below → U_above)
        T_D: down-going transmission (D_above → D_below)
    """
    nfreq = R_above.shape[0]

    Ra_D = R_above[:, :, _D_COLS]  # (nfreq, 4, 2) - D columns of above
    Ra_U = R_above[:, :, _U_COLS]  # (nfreq, 4, 2) - U columns of above
    Rb_D = R_below[:, :, _D_COLS]  # (nfreq, 4, 2) - D columns of below
    Rb_U = R_below[:, :, _U_COLS]  # (nfreq, 4, 2) - U columns of below

    # --- Up-going from below ---
    # N @ [U_above; D_below] = Rb_U @ U_below
    # N = [Ra_U | -Rb_D], shape (nfreq, 4, 4)
    N = np.concatenate([Ra_U, -Rb_D], axis=2)
    # Solve: [U_above; D_below] = N^{-1} @ Rb_U
    sol_up = np.linalg.solve(N, Rb_U)  # (nfreq, 4, 2)
    T_U = sol_up[:, :2, :]  # first 2 rows: U_above
    R_U = sol_up[:, 2:, :]  # last 2 rows: D_below

    # --- Down-going from above ---
    # M @ [D_below; U_above] = Ra_D @ D_above
    # M = [Rb_D | -Ra_U], shape (nfreq, 4, 4)
    M = np.concatenate([Rb_D, -Ra_U], axis=2)
    sol_dn = np.linalg.solve(M, Ra_D)  # (nfreq, 4, 2)
    T_D = sol_dn[:, :2, :]  # first 2 rows: D_below
    R_D = sol_dn[:, 2:, :]  # last 2 rows: U_above

    return R_U, R_D, T_U, T_D


def _phase_delay(omega, p_si, vp, vs, thickness):
    """One-way phase delay matrix E for propagation through a layer.

    E = diag(exp(-i*ksz*h), exp(-i*kpz*h)) in [S, P] ordering.

    Returns
    -------
    E : ndarray, shape (nfreq, 2, 2)
        Diagonal phase delay matrix.
    kpz, ksz : ndarray, shape (nfreq,)
        Vertical wavenumbers.
    """
    kh, kpz, ksz, gamma = vertical_wavenumbers(omega, p_si, vp, vs)
    nfreq = kh.shape[0]
    E = np.zeros((nfreq, 2, 2), dtype=complex)
    E[:, 0, 0] = np.exp(-1j * ksz * thickness)
    E[:, 1, 1] = np.exp(-1j * kpz * thickness)
    return E, kpz, ksz


def _kennett_recursion(model: FKLayerModel, omega, p_si):
    """Run the Kennett recursion from the free surface to the half-space.

    Computes generalized reflection G_j and related quantities for each layer.

    Parameters
    ----------
    model : FKLayerModel
    omega : ndarray, shape (nfreq,)
    p_si : float

    Returns
    -------
    G_list : list of ndarray (nfreq, 2, 2)
        G_list[j] = generalized reflection at top of layer j.
    Q_list : list of ndarray (nfreq, 2, 2)
        Q_list[j] = E_j @ G_j @ E_j (reverberant reflector seen from bottom of layer j).
    R_D_list : list of ndarray (nfreq, 2, 2)
        R_D_list[j] = R_D at interface j+1.
    T_U_list : list of ndarray (nfreq, 2, 2)
        T_U_list[j] = T_U at interface j+1.
    E_list : list of ndarray (nfreq, 2, 2)
        E_list[j] = phase delay through layer j.
    R_eigen_list : list of ndarray (nfreq, 4, 4)
        Eigenvector matrices for each layer.
    """
    nfreq = omega.shape[0]
    n_entries = model.n_entries
    n_finite = model.n_finite_layers
    th = model.thickness

    # Pre-compute eigenvector matrices for all layers.
    R_eigen_list = []
    kpz_list = []
    ksz_list = []
    for i in range(n_entries):
        R_eig, kpz, ksz = _eigen_R_at_interface(
            omega, p_si, model.vp[i], model.vs[i], model.rho[i]
        )
        R_eigen_list.append(R_eig)
        kpz_list.append(kpz)
        ksz_list.append(ksz)

    # Free-surface reflection.
    R_F = _free_surface_reflection(R_eigen_list[0])

    # Phase delays for finite layers.
    E_list = []
    for j in range(n_finite):
        E, _, _ = _phase_delay(omega, p_si, model.vp[j], model.vs[j], th[j])
        E_list.append(E)

    # Forward recursion: build G_j from top to bottom.
    G_list = [None] * (n_finite + 1)
    Q_list = [None] * n_finite
    R_D_list = [None] * n_finite
    T_U_list = [None] * n_finite

    G_list[0] = R_F

    for j in range(n_finite):
        E_j = E_list[j]
        # Q_j = E_j @ G_j @ E_j
        Q_j = E_j @ G_list[j] @ E_j
        Q_list[j] = Q_j

        # Interface j+1 R/T (between layer j above and layer j+1 below).
        R_U, R_D, T_U, T_D = _interface_rt(R_eigen_list[j], R_eigen_list[j + 1])
        R_D_list[j] = R_D
        T_U_list[j] = T_U

        # Reverberation operator: [I - R_D @ Q_j]^{-1}
        I2 = np.broadcast_to(np.eye(2, dtype=complex), (nfreq, 2, 2)).copy()
        Rev = np.linalg.inv(I2 - R_D @ Q_j)

        # Addition rule.
        G_list[j + 1] = R_U + T_D @ Q_j @ Rev @ T_U

    return G_list, Q_list, R_D_list, T_U_list, E_list, R_eigen_list


def _incident_vector(omega, p_si, model, incidence, z_init, amp):
    """Construct the 2x2 incident up-going amplitude vector in the half-space.

    Includes phase delay from z_init to the top of the half-space.

    Returns
    -------
    u_inc : ndarray, shape (nfreq, 2)
        Incident amplitudes in [S, P] ordering.
    """
    nfreq = omega.shape[0]
    half = model.halfspace_index
    kh, kpz_hs, ksz_hs, _ = vertical_wavenumbers(
        omega, p_si, model.vp[half], model.vs[half]
    )
    dz_init = z_init - model.z_top[half]  # positive (z_init deeper than z_top_hs)

    u_inc = np.zeros((nfreq, 2), dtype=complex)
    incidence = incidence.upper()
    if incidence == "P":
        phase = np.exp(-1j * kpz_hs * dz_init)
        u_inc[:, 1] = amp * phase  # P component
    elif incidence == "S":
        phase = np.exp(-1j * ksz_hs * dz_init)
        u_inc[:, 0] = amp * phase  # S component
    else:
        raise ValueError("incidence must be 'P' or 'S'.")
    return u_inc


def surface_state_kennett(model: FKLayerModel, omega, p_si, incidence,
                          z_init, amp=None):
    """Free-surface state vector using Kennett's reflectivity method.

    Returns
    -------
    y0 : ndarray, shape (nfreq, 4)
        State vector [ux, uz, sigma_hz, sigma_zz] at z=0.
        (sigma_hz = sigma_zz = 0 at the free surface.)
    """
    omega = np.asarray(omega, dtype=float)
    nfreq = omega.shape[0]

    if amp is None:
        half = model.halfspace_index
        v = model.vp[half] if incidence.upper() == "P" else model.vs[half]
        amp = float(np.sin(incidence_angle(p_si, v)))

    # Run Kennett recursion.
    G_list, Q_list, R_D_list, T_U_list, E_list, R_eigen_list = \
        _kennett_recursion(model, omega, p_si)

    # Incident wave in half-space.
    u_inc = _incident_vector(omega, p_si, model, incidence, z_init, amp)

    # Backward propagation: compute u at top of each layer (from halfspace up).
    n_finite = model.n_finite_layers
    u_top = [None] * (n_finite + 1)
    u_top[n_finite] = u_inc  # at top of halfspace

    for j in range(n_finite - 1, -1, -1):
        E_j = E_list[j]
        Q_j = Q_list[j]
        R_D_j = R_D_list[j]
        T_U_j = T_U_list[j]
        I2 = np.broadcast_to(np.eye(2, dtype=complex), (nfreq, 2, 2)).copy()
        Rev = np.linalg.inv(I2 - R_D_j @ Q_j)
        u_top[j] = np.einsum(
            "fij,fj->fi", E_j @ Rev @ T_U_j, u_top[j + 1]
        )

    # At the free surface: c_0 = [DS, US, DP, UP]
    u_0 = u_top[0]  # up-going at surface, [S, P]
    d_0 = np.einsum("fij,fj->fi", G_list[0], u_0)  # down-going at surface

    c = np.zeros((nfreq, 4), dtype=complex)
    c[:, IDX_DS] = d_0[:, 0]  # S component of down-going
    c[:, IDX_US] = u_0[:, 0]  # S component of up-going
    c[:, IDX_DP] = d_0[:, 1]  # P component of down-going
    c[:, IDX_UP] = u_0[:, 1]  # P component of up-going

    # State vector y = R_0 @ c
    R0 = R_eigen_list[0]
    y0 = np.einsum("fij,fj->fi", R0, c)
    return y0


def state_at_depth_kennett(model: FKLayerModel, omega, p_si, incidence,
                           z_init, z_r, amp=None):
    """State vector at arbitrary depth z_r using Kennett's reflectivity method.

    Returns
    -------
    y : ndarray, shape (nfreq, 4)
        State vector [ux, uz, sigma_hz, sigma_zz] at depth z_r.
    """
    omega = np.asarray(omega, dtype=float)
    nfreq = omega.shape[0]

    if amp is None:
        half = model.halfspace_index
        v = model.vp[half] if incidence.upper() == "P" else model.vs[half]
        amp = float(np.sin(incidence_angle(p_si, v)))

    # Run Kennett recursion.
    G_list, Q_list, R_D_list, T_U_list, E_list, R_eigen_list = \
        _kennett_recursion(model, omega, p_si)

    # Incident wave.
    u_inc = _incident_vector(omega, p_si, model, incidence, z_init, amp)

    # Backward propagation to get u at top of each layer.
    n_finite = model.n_finite_layers
    u_top = [None] * (n_finite + 1)
    u_top[n_finite] = u_inc

    for j in range(n_finite - 1, -1, -1):
        E_j = E_list[j]
        Q_j = Q_list[j]
        R_D_j = R_D_list[j]
        T_U_j = T_U_list[j]
        I2 = np.broadcast_to(np.eye(2, dtype=complex), (nfreq, 2, 2)).copy()
        Rev = np.linalg.inv(I2 - R_D_j @ Q_j)
        u_top[j] = np.einsum(
            "fij,fj->fi", E_j @ Rev @ T_U_j, u_top[j + 1]
        )

    # Determine which layer contains z_r.
    layer_idx = _layer_index_of_depth(model, z_r)

    # Amplitudes at top of the target layer.
    if layer_idx <= n_finite - 1:
        u_k = u_top[layer_idx]
    else:
        u_k = u_top[n_finite]  # halfspace

    d_k = np.einsum("fij,fj->fi", G_list[layer_idx], u_k)

    # Phase to depth z_r within the layer.
    dz = z_r - model.z_top[layer_idx]
    kh, kpz, ksz, _ = vertical_wavenumbers(
        omega, p_si, model.vp[layer_idx], model.vs[layer_idx]
    )

    # Coefficients at depth z_r: apply H(dz) component-wise.
    c = np.zeros((nfreq, 4), dtype=complex)
    c[:, IDX_DS] = np.exp(-1j * ksz * dz) * d_k[:, 0]
    c[:, IDX_US] = np.exp(1j * ksz * dz) * u_k[:, 0]
    c[:, IDX_DP] = np.exp(-1j * kpz * dz) * d_k[:, 1]
    c[:, IDX_UP] = np.exp(1j * kpz * dz) * u_k[:, 1]

    # State vector y = R_layer @ c
    R_k = R_eigen_list[layer_idx]
    y = np.einsum("fij,fj->fi", R_k, c)
    return y


def _layer_index_of_depth(model: FKLayerModel, z: float) -> int:
    """Entry index containing depth z (half-space for z >= last z_top)."""
    if z <= model.z_top[0]:
        return 0
    idx = int(np.searchsorted(model.z_top, z, side="right") - 1)
    return min(idx, model.halfspace_index)
