"""Numba-accelerated Kennett reflectivity for P-SV plane-wave response.

Parallelises the Kennett recursive reflectivity algorithm over frequency
using ``@njit(parallel=True, cache=True)`` with ``prange``. Each frequency
is handled independently, making this embarrassingly parallel.

Public wrappers (``surface_state_kennett_numba``, ``states_at_depths_kennett_numba``)
accept the same ``FKLayerModel`` objects as :mod:`fk.propagators.kennett` and return
identically shaped arrays, so they are drop-in replacements.

State vector: y = [ux, uz, sigma_hz, sigma_zz]
2x2 sub-system ordering: [S, P] (index 0 = S, index 1 = P).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..geometry import incidence_angle
from ..model import FKLayerModel

# Coefficient column indices matching haskell.py.
IDX_DS, IDX_US, IDX_DP, IDX_UP = 0, 1, 2, 3


# ------------------------------------------------------------------ #
# Low-level Numba helpers (single-frequency, scalar operations)
# ------------------------------------------------------------------ #

@njit(cache=True)
def _build_R_single(w, p_si, vp, vs, rho):
    """Eigenvector matrix R (4x4) and vertical wavenumbers for one frequency."""
    kh = w * p_si
    kpz = np.sqrt((w / vp) ** 2 - kh ** 2 + 0.0j)
    ksz = np.sqrt((w / vs) ** 2 - kh ** 2 + 0.0j)
    gamma = (w / vs) ** 2 - 2.0 * kh ** 2
    mu = rho * vs ** 2

    ks_kh = ksz / kh
    kp_kh = kpz / kh
    img = 1.0j

    R = np.zeros((4, 4), dtype=np.complex128)
    R[0, 0] = ks_kh;   R[0, 1] = -ks_kh;  R[0, 2] = 1.0;     R[0, 3] = 1.0
    R[1, 0] = 1.0;     R[1, 1] = 1.0;     R[1, 2] = -kp_kh;  R[1, 3] = kp_kh
    R[2, 0] = -img * mu * gamma / kh
    R[2, 1] = -img * mu * gamma / kh
    R[2, 2] = -2.0 * img * mu * kpz
    R[2, 3] = 2.0 * img * mu * kpz
    R[3, 0] = -2.0 * img * mu * ksz
    R[3, 1] = 2.0 * img * mu * ksz
    R[3, 2] = img * mu * gamma / kh
    R[3, 3] = img * mu * gamma / kh
    return R, kpz, ksz


@njit(cache=True)
def _mat2_mul(A, B):
    """2x2 complex matrix multiply."""
    C = np.zeros((2, 2), dtype=np.complex128)
    C[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
    C[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
    C[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
    C[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
    return C


@njit(cache=True)
def _mat2_vec(A, v):
    """2x2 matrix times length-2 vector."""
    out = np.zeros(2, dtype=np.complex128)
    out[0] = A[0, 0] * v[0] + A[0, 1] * v[1]
    out[1] = A[1, 0] * v[0] + A[1, 1] * v[1]
    return out


@njit(cache=True)
def _mat2_inv(A):
    """2x2 complex matrix inverse using explicit formula."""
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    Ainv = np.zeros((2, 2), dtype=np.complex128)
    Ainv[0, 0] = A[1, 1] / det
    Ainv[0, 1] = -A[0, 1] / det
    Ainv[1, 0] = -A[1, 0] / det
    Ainv[1, 1] = A[0, 0] / det
    return Ainv


@njit(cache=True)
def _mat4_vec(A, v):
    """4x4 matrix times length-4 vector."""
    out = np.zeros(4, dtype=np.complex128)
    for i in range(4):
        s = 0.0 + 0.0j
        for j in range(4):
            s += A[i, j] * v[j]
        out[i] = s
    return out


@njit(cache=True)
def _solve4x4(A, B):
    """Solve A @ X = B for X, where A is 4x4 and B is 4x2.

    Uses partial pivoting Gaussian elimination.
    """
    n = 4
    ncols = 2
    # Augmented matrix [A | B]
    aug = np.zeros((n, n + ncols), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            aug[i, j] = A[i, j]
        for j in range(ncols):
            aug[i, n + j] = B[i, j]

    # Forward elimination with partial pivoting.
    for col in range(n):
        # Find pivot.
        max_val = 0.0
        max_row = col
        for row in range(col, n):
            val = abs(aug[row, col])
            if val > max_val:
                max_val = val
                max_row = row
        # Swap rows.
        if max_row != col:
            for j in range(n + ncols):
                tmp = aug[col, j]
                aug[col, j] = aug[max_row, j]
                aug[max_row, j] = tmp
        # Eliminate below.
        pivot = aug[col, col]
        for row in range(col + 1, n):
            factor = aug[row, col] / pivot
            for j in range(col, n + ncols):
                aug[row, j] -= factor * aug[col, j]

    # Back substitution.
    X = np.zeros((n, ncols), dtype=np.complex128)
    for col_b in range(ncols):
        for row in range(n - 1, -1, -1):
            s = aug[row, n + col_b]
            for j in range(row + 1, n):
                s -= aug[row, j] * X[j, col_b]
            X[row, col_b] = s / aug[row, row]
    return X


@njit(cache=True)
def _free_surface_refl(R0):
    """Free-surface reflection R_F (2x2): maps U=[US,UP] → D=[DS,DP].

    From zero-traction BC: R0[2:4, D_cols] @ D + R0[2:4, U_cols] @ U = 0
    => D = -inv(stress_D) @ stress_U @ U
    """
    # stress_D = R0[2:4, [0, 2]] (columns for DS, DP)
    stress_D = np.zeros((2, 2), dtype=np.complex128)
    stress_D[0, 0] = R0[2, 0]  # row 2, DS
    stress_D[0, 1] = R0[2, 2]  # row 2, DP
    stress_D[1, 0] = R0[3, 0]  # row 3, DS
    stress_D[1, 1] = R0[3, 2]  # row 3, DP

    # stress_U = R0[2:4, [1, 3]] (columns for US, UP)
    stress_U = np.zeros((2, 2), dtype=np.complex128)
    stress_U[0, 0] = R0[2, 1]  # row 2, US
    stress_U[0, 1] = R0[2, 3]  # row 2, UP
    stress_U[1, 0] = R0[3, 1]  # row 3, US
    stress_U[1, 1] = R0[3, 3]  # row 3, UP

    inv_stress_D = _mat2_inv(stress_D)
    R_F = np.zeros((2, 2), dtype=np.complex128)
    # R_F = -inv_stress_D @ stress_U
    for i in range(2):
        for j in range(2):
            s = 0.0 + 0.0j
            for k in range(2):
                s += inv_stress_D[i, k] * stress_U[k, j]
            R_F[i, j] = -s
    return R_F


@njit(cache=True)
def _interface_rt_single(R_above, R_below):
    """Compute 2x2 R/T at one interface for one frequency.

    Returns R_U, R_D, T_U, T_D (each 2x2).
    """
    # Extract D and U columns.
    # D_cols = [0, 2], U_cols = [1, 3]
    Ra_U = np.zeros((4, 2), dtype=np.complex128)
    Ra_D = np.zeros((4, 2), dtype=np.complex128)
    Rb_U = np.zeros((4, 2), dtype=np.complex128)
    Rb_D = np.zeros((4, 2), dtype=np.complex128)
    for i in range(4):
        Ra_D[i, 0] = R_above[i, 0]  # DS
        Ra_D[i, 1] = R_above[i, 2]  # DP
        Ra_U[i, 0] = R_above[i, 1]  # US
        Ra_U[i, 1] = R_above[i, 3]  # UP
        Rb_D[i, 0] = R_below[i, 0]
        Rb_D[i, 1] = R_below[i, 2]
        Rb_U[i, 0] = R_below[i, 1]
        Rb_U[i, 1] = R_below[i, 3]

    # Up-going from below: N @ [U_a; D_b] = Rb_U @ U_b
    # N = [Ra_U | -Rb_D] (4x4)
    N = np.zeros((4, 4), dtype=np.complex128)
    for i in range(4):
        N[i, 0] = Ra_U[i, 0]
        N[i, 1] = Ra_U[i, 1]
        N[i, 2] = -Rb_D[i, 0]
        N[i, 3] = -Rb_D[i, 1]
    sol_up = _solve4x4(N, Rb_U)  # (4, 2)
    T_U = np.zeros((2, 2), dtype=np.complex128)
    R_U = np.zeros((2, 2), dtype=np.complex128)
    T_U[0, 0] = sol_up[0, 0]; T_U[0, 1] = sol_up[0, 1]
    T_U[1, 0] = sol_up[1, 0]; T_U[1, 1] = sol_up[1, 1]
    R_U[0, 0] = sol_up[2, 0]; R_U[0, 1] = sol_up[2, 1]
    R_U[1, 0] = sol_up[3, 0]; R_U[1, 1] = sol_up[3, 1]

    # Down-going from above: M @ [D_b; U_a] = Ra_D @ D_a
    # M = [Rb_D | -Ra_U] (4x4)
    M = np.zeros((4, 4), dtype=np.complex128)
    for i in range(4):
        M[i, 0] = Rb_D[i, 0]
        M[i, 1] = Rb_D[i, 1]
        M[i, 2] = -Ra_U[i, 0]
        M[i, 3] = -Ra_U[i, 1]
    sol_dn = _solve4x4(M, Ra_D)  # (4, 2)
    T_D = np.zeros((2, 2), dtype=np.complex128)
    R_D = np.zeros((2, 2), dtype=np.complex128)
    T_D[0, 0] = sol_dn[0, 0]; T_D[0, 1] = sol_dn[0, 1]
    T_D[1, 0] = sol_dn[1, 0]; T_D[1, 1] = sol_dn[1, 1]
    R_D[0, 0] = sol_dn[2, 0]; R_D[0, 1] = sol_dn[2, 1]
    R_D[1, 0] = sol_dn[3, 0]; R_D[1, 1] = sol_dn[3, 1]

    return R_U, R_D, T_U, T_D


# ------------------------------------------------------------------ #
# Main parallel kernel
# ------------------------------------------------------------------ #

@njit(parallel=True, cache=True)
def _kennett_states_kernel(omega, p_si, vp, vs, rho, z_top,
                           thickness, n_finite, z_init,
                           inc_col, amp, z_depths, layer_indices):
    """Compute state vectors at multiple depths using Kennett reflectivity.

    Parallelises over frequency with prange.

    Parameters
    ----------
    omega : float64 array (nfreq,)
    p_si : float64
    vp, vs, rho, z_top, thickness : float64 arrays (n_entries,)
    n_finite : int
    z_init : float64
    inc_col : int (0 for S incidence [S index], 1 for P incidence [P index])
    amp : float64
    z_depths : float64 array (n_depths,) sorted ascending
    layer_indices : int64 array (n_depths,)

    Returns
    -------
    result : complex128 array (nfreq, n_depths, 4)
    """
    nfreq = omega.shape[0]
    n_depths = z_depths.shape[0]
    n_entries = vp.shape[0]
    result = np.zeros((nfreq, n_depths, 4), dtype=np.complex128)

    for f in prange(nfreq):
        w = omega[f]

        # --- Pre-compute eigenvector matrices for all layers ---
        R_eigen = np.zeros((n_entries, 4, 4), dtype=np.complex128)
        kpz_arr = np.zeros(n_entries, dtype=np.complex128)
        ksz_arr = np.zeros(n_entries, dtype=np.complex128)
        for i in range(n_entries):
            R_i, kpz_i, ksz_i = _build_R_single(w, p_si, vp[i], vs[i], rho[i])
            R_eigen[i] = R_i
            kpz_arr[i] = kpz_i
            ksz_arr[i] = ksz_i

        # --- Free-surface reflection ---
        R_F = _free_surface_refl(R_eigen[0])

        # --- Forward recursion: build G_j ---
        # Store G and Q for each layer.
        G = np.zeros((n_finite + 1, 2, 2), dtype=np.complex128)
        Q = np.zeros((n_finite, 2, 2), dtype=np.complex128)
        R_D_store = np.zeros((n_finite, 2, 2), dtype=np.complex128)
        T_U_store = np.zeros((n_finite, 2, 2), dtype=np.complex128)
        E_store = np.zeros((n_finite, 2, 2), dtype=np.complex128)

        G[0] = R_F

        for j in range(n_finite):
            # Phase delay through layer j.
            E_j = np.zeros((2, 2), dtype=np.complex128)
            E_j[0, 0] = np.exp(-1.0j * ksz_arr[j] * thickness[j])
            E_j[1, 1] = np.exp(-1.0j * kpz_arr[j] * thickness[j])
            E_store[j] = E_j

            # Q_j = E_j @ G[j] @ E_j
            Q_j = _mat2_mul(_mat2_mul(E_j, G[j]), E_j)
            Q[j] = Q_j

            # Interface R/T between layer j (above) and layer j+1 (below).
            R_U_j, R_D_j, T_U_j, T_D_j = _interface_rt_single(
                R_eigen[j], R_eigen[j + 1]
            )
            R_D_store[j] = R_D_j
            T_U_store[j] = T_U_j

            # Reverberation: Rev = [I - R_D @ Q_j]^{-1}
            I2 = np.zeros((2, 2), dtype=np.complex128)
            I2[0, 0] = 1.0; I2[1, 1] = 1.0
            RD_Q = _mat2_mul(R_D_j, Q_j)
            Rev = _mat2_inv(I2 - RD_Q)

            # Addition rule: G[j+1] = R_U + T_D @ Q_j @ Rev @ T_U
            G[j + 1] = R_U_j + _mat2_mul(
                _mat2_mul(_mat2_mul(T_D_j, Q_j), Rev), T_U_j
            )

        # --- Incident wave in half-space ---
        half = n_entries - 1
        dz_init = z_init - z_top[half]
        u_inc = np.zeros(2, dtype=np.complex128)
        if inc_col == 1:  # P incidence
            u_inc[1] = amp * np.exp(-1.0j * kpz_arr[half] * dz_init)
        else:  # S incidence
            u_inc[0] = amp * np.exp(-1.0j * ksz_arr[half] * dz_init)

        # --- Backward propagation: u at top of each layer ---
        u_top = np.zeros((n_finite + 1, 2), dtype=np.complex128)
        u_top[n_finite] = u_inc

        for j in range(n_finite - 1, -1, -1):
            I2 = np.zeros((2, 2), dtype=np.complex128)
            I2[0, 0] = 1.0; I2[1, 1] = 1.0
            RD_Q = _mat2_mul(R_D_store[j], Q[j])
            Rev = _mat2_inv(I2 - RD_Q)
            # u_top[j] = E @ Rev @ T_U @ u_top[j+1]
            tmp = _mat2_vec(T_U_store[j], u_top[j + 1])
            tmp = _mat2_vec(Rev, tmp)
            u_top[j] = _mat2_vec(E_store[j], tmp)

        # --- Compute state at each requested depth ---
        for d in range(n_depths):
            li = layer_indices[d]
            z_r = z_depths[d]
            dz = z_r - z_top[li]

            # Get amplitudes at top of target layer.
            u_k = np.zeros(2, dtype=np.complex128)
            if li <= n_finite - 1:
                u_k[0] = u_top[li, 0]
                u_k[1] = u_top[li, 1]
            else:
                u_k[0] = u_top[n_finite, 0]
                u_k[1] = u_top[n_finite, 1]

            # Down-going at top of target layer: d_k = G[li] @ u_k
            d_k = _mat2_vec(G[li], u_k)

            # Phase to depth z_r.
            exp_ks = np.exp(-1.0j * ksz_arr[li] * dz)
            exp_kp = np.exp(-1.0j * kpz_arr[li] * dz)

            # Coefficients at z_r: c = [DS, US, DP, UP]
            c = np.zeros(4, dtype=np.complex128)
            c[0] = exp_ks * d_k[0]        # DS: exp(-i*ksz*dz) * d[S]
            c[1] = (1.0 / exp_ks) * u_k[0]  # US: exp(+i*ksz*dz) * u[S]
            c[2] = exp_kp * d_k[1]        # DP: exp(-i*kpz*dz) * d[P]
            c[3] = (1.0 / exp_kp) * u_k[1]  # UP: exp(+i*kpz*dz) * u[P]

            # State vector: y = R_layer @ c
            y = _mat4_vec(R_eigen[li], c)
            result[f, d, 0] = y[0]
            result[f, d, 1] = y[1]
            result[f, d, 2] = y[2]
            result[f, d, 3] = y[3]

    return result


# ------------------------------------------------------------------ #
# Preparation helper (pure Python, runs once)
# ------------------------------------------------------------------ #

def _prepare_depth_args(model: FKLayerModel, z_depths):
    """Sort depths and compute their layer indices."""
    z_depths = np.atleast_1d(np.asarray(z_depths, dtype=np.float64))
    order = np.argsort(z_depths)
    z_sorted = z_depths[order]
    layer_indices = np.searchsorted(model.z_top, z_sorted, side="right") - 1
    layer_indices = np.clip(layer_indices, 0, model.halfspace_index)
    layer_indices[z_sorted <= model.z_top[0]] = 0
    return z_sorted, layer_indices.astype(np.int64), order


# ------------------------------------------------------------------ #
# Public wrappers
# ------------------------------------------------------------------ #

def surface_state_kennett_numba(model: FKLayerModel, omega, p_si, incidence,
                                z_init, amp=None):
    """Kennett reflectivity surface state (Numba-accelerated).

    Drop-in replacement for :func:`fk.propagators.kennett.surface_state_kennett`.

    Returns
    -------
    y0 : ndarray, shape (nfreq, 4)
        State vector [ux, uz, sigma_hz, sigma_zz] at z=0.
    """
    if amp is None:
        half = model.halfspace_index
        v = model.vp[half] if incidence.upper() == "P" else model.vs[half]
        amp = float(np.sin(incidence_angle(p_si, v)))
    inc_col = 1 if incidence.upper() == "P" else 0  # [S, P] ordering
    omega = np.asarray(omega, dtype=np.float64)
    z_depths = np.array([0.0], dtype=np.float64)
    layer_indices = np.array([0], dtype=np.int64)
    res = _kennett_states_kernel(
        omega, float(p_si),
        model.vp.astype(np.float64), model.vs.astype(np.float64),
        model.rho.astype(np.float64), model.z_top.astype(np.float64),
        model.thickness.astype(np.float64),
        model.n_finite_layers, float(z_init),
        inc_col, float(amp), z_depths, layer_indices,
    )
    return res[:, 0, :]  # (nfreq, 4)


def states_at_depths_kennett_numba(model: FKLayerModel, omega, p_si, incidence,
                                   z_init, z_depths, amp=None):
    """Batch-compute state vectors at multiple depths using Kennett (Numba).

    Drop-in replacement for calling ``state_at_depth_kennett`` in a loop,
    and compatible with ``states_at_depths_numba`` for hybrid wavefield injection.

    Returns
    -------
    result : ndarray, shape (nfreq, n_depths, 4)
        State vectors in the *original* order of ``z_depths``.
    """
    if amp is None:
        half = model.halfspace_index
        v = model.vp[half] if incidence.upper() == "P" else model.vs[half]
        amp = float(np.sin(incidence_angle(p_si, v)))
    inc_col = 1 if incidence.upper() == "P" else 0
    omega = np.asarray(omega, dtype=np.float64)

    z_sorted, layer_idx, order = _prepare_depth_args(model, z_depths)

    res = _kennett_states_kernel(
        omega, float(p_si),
        model.vp.astype(np.float64), model.vs.astype(np.float64),
        model.rho.astype(np.float64), model.z_top.astype(np.float64),
        model.thickness.astype(np.float64),
        model.n_finite_layers, float(z_init),
        inc_col, float(amp), z_sorted, layer_idx,
    )
    inv_order = np.argsort(order)
    return res[:, inv_order, :]
