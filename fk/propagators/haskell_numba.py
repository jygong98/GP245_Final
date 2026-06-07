"""Numba-accelerated propagator-matrix kernels for hybrid-stage FK workloads.

Fuses the build-propagator / free-surface-BC-solve / downward-continuation
pipeline into a single ``@njit(parallel=True)`` kernel that parallelises over
frequency with ``prange``.  The per-frequency work is fully inlined: 4x4
matrix assembly, Cramer's-rule 2x2 solve, and cascaded depth continuation —
eliminating repeated Python-level calls and LAPACK dispatch overhead that
dominate at O(nz) boundary depths.

The public wrappers (``surface_state_numba``, ``states_at_depths_numba``)
accept the same ``FKLayerModel`` objects as :mod:`fk.propagators.haskell` and return
identically shaped arrays, so they are drop-in replacements.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..geometry import incidence_angle
from ..model import FKLayerModel

# Coefficient column indices — must match haskell.py.
IDX_DS, IDX_US, IDX_DP, IDX_UP = 0, 1, 2, 3


# ------------------------------------------------------------------ #
# Low-level helpers (single-frequency, single-element)
# ------------------------------------------------------------------ #

@njit(cache=True)
def _inv4x4(A):
    """4x4 complex matrix inverse using the sub-determinant method.

    Computes 2x2 minors from the top and bottom row-pairs, then
    assembles the adjugate and determinant.  Much faster than
    np.linalg.inv inside @njit for small fixed-size matrices because
    it avoids generic dispatch and temporary array allocation.
    """
    s0 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    s1 = A[0, 0] * A[1, 2] - A[0, 2] * A[1, 0]
    s2 = A[0, 0] * A[1, 3] - A[0, 3] * A[1, 0]
    s3 = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    s4 = A[0, 1] * A[1, 3] - A[0, 3] * A[1, 1]
    s5 = A[0, 2] * A[1, 3] - A[0, 3] * A[1, 2]

    c5 = A[2, 2] * A[3, 3] - A[2, 3] * A[3, 2]
    c4 = A[2, 1] * A[3, 3] - A[2, 3] * A[3, 1]
    c3 = A[2, 1] * A[3, 2] - A[2, 2] * A[3, 1]
    c2 = A[2, 0] * A[3, 3] - A[2, 3] * A[3, 0]
    c1 = A[2, 0] * A[3, 2] - A[2, 2] * A[3, 0]
    c0 = A[2, 0] * A[3, 1] - A[2, 1] * A[3, 0]

    idet = 1.0 / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0)

    B = np.empty((4, 4), dtype=np.complex128)
    B[0, 0] = ( A[1, 1] * c5 - A[1, 2] * c4 + A[1, 3] * c3) * idet
    B[0, 1] = (-A[0, 1] * c5 + A[0, 2] * c4 - A[0, 3] * c3) * idet
    B[0, 2] = ( A[3, 1] * s5 - A[3, 2] * s4 + A[3, 3] * s3) * idet
    B[0, 3] = (-A[2, 1] * s5 + A[2, 2] * s4 - A[2, 3] * s3) * idet

    B[1, 0] = (-A[1, 0] * c5 + A[1, 2] * c2 - A[1, 3] * c1) * idet
    B[1, 1] = ( A[0, 0] * c5 - A[0, 2] * c2 + A[0, 3] * c1) * idet
    B[1, 2] = (-A[3, 0] * s5 + A[3, 2] * s2 - A[3, 3] * s1) * idet
    B[1, 3] = ( A[2, 0] * s5 - A[2, 2] * s2 + A[2, 3] * s1) * idet

    B[2, 0] = ( A[1, 0] * c4 - A[1, 1] * c2 + A[1, 3] * c0) * idet
    B[2, 1] = (-A[0, 0] * c4 + A[0, 1] * c2 - A[0, 3] * c0) * idet
    B[2, 2] = ( A[3, 0] * s4 - A[3, 1] * s2 + A[3, 3] * s0) * idet
    B[2, 3] = (-A[2, 0] * s4 + A[2, 1] * s2 - A[2, 3] * s0) * idet

    B[3, 0] = (-A[1, 0] * c3 + A[1, 1] * c1 - A[1, 2] * c0) * idet
    B[3, 1] = ( A[0, 0] * c3 - A[0, 1] * c1 + A[0, 2] * c0) * idet
    B[3, 2] = (-A[3, 0] * s3 + A[3, 1] * s1 - A[3, 2] * s0) * idet
    B[3, 3] = ( A[2, 0] * s3 - A[2, 1] * s1 + A[2, 2] * s0) * idet
    return B


@njit(cache=True)
def _mat4_mul(A, B):
    """4x4 complex matrix multiply."""
    C = np.zeros((4, 4), dtype=np.complex128)
    for i in range(4):
        for j in range(4):
            s = 0.0 + 0.0j
            for k in range(4):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


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
def _build_R(w, p_si, vp_i, vs_i, rho_i):
    """Eigenvector matrix R (4x4) for one layer at one frequency."""
    kh = w * p_si
    kp_z = np.sqrt((w / vp_i) ** 2 - kh ** 2 + 0.0j)
    ks_z = np.sqrt((w / vs_i) ** 2 - kh ** 2 + 0.0j)
    gamma = (w / vs_i) ** 2 - 2.0 * kh ** 2
    mu = rho_i * vs_i ** 2

    ks_kh = ks_z / kh
    kp_kh = kp_z / kh
    img = 1.0j

    R = np.zeros((4, 4), dtype=np.complex128)
    R[0, 0] = ks_kh;   R[0, 1] = -ks_kh;  R[0, 2] = 1.0;     R[0, 3] = 1.0
    R[1, 0] = 1.0;     R[1, 1] = 1.0;     R[1, 2] = -kp_kh;  R[1, 3] = kp_kh
    R[2, 0] = -img * mu * gamma / kh
    R[2, 1] = -img * mu * gamma / kh
    R[2, 2] = -2.0 * img * mu * kp_z
    R[2, 3] = 2.0 * img * mu * kp_z
    R[3, 0] = -2.0 * img * mu * ks_z
    R[3, 1] = 2.0 * img * mu * ks_z
    R[3, 2] = img * mu * gamma / kh
    R[3, 3] = img * mu * gamma / kh
    return R, kp_z, ks_z


@njit(cache=True)
def _build_H(kp_z, ks_z, dz):
    """Phase matrix H (4x4 diagonal) at one frequency."""
    H = np.zeros((4, 4), dtype=np.complex128)
    H[0, 0] = np.exp(-1.0j * ks_z * dz)
    H[1, 1] = np.exp(1.0j * ks_z * dz)
    H[2, 2] = np.exp(-1.0j * kp_z * dz)
    H[3, 3] = np.exp(1.0j * kp_z * dz)
    return H


@njit(cache=True)
def _layer_prop(w, p_si, vp_i, vs_i, rho_i, thickness):
    """Layer propagator P = R H(-h) R^-1 for one layer at one frequency."""
    R, kp_z, ks_z = _build_R(w, p_si, vp_i, vs_i, rho_i)
    H = _build_H(kp_z, ks_z, -thickness)
    Rinv = _inv4x4(R)
    return _mat4_mul(_mat4_mul(R, H), Rinv)


# ------------------------------------------------------------------ #
# Main parallel kernel
# ------------------------------------------------------------------ #

@njit(parallel=True, cache=True)
def _boundary_states_kernel(omega, p_si, vp, vs, rho, z_top,
                            thickness, n_finite, z_init,
                            inc_col, amp, z_depths, layer_indices):
    """Compute state vectors y(z_r) for all (frequency, depth) pairs.

    Parameters
    ----------
    omega : float64 array (nfreq,)
    p_si : float64
    vp, vs, rho, z_top, thickness : float64 arrays (n_entries,)
    n_finite : int
    z_init : float64
    inc_col : int  (IDX_UP for P, IDX_US for S)
    amp : float64
    z_depths : float64 array (n_depths,)  — must be sorted ascending
    layer_indices : int64 array (n_depths,)

    Returns
    -------
    result : complex128 array (nfreq, n_depths, 4)
    """
    nfreq = omega.shape[0]
    n_depths = z_depths.shape[0]
    half = vp.shape[0] - 1
    result = np.zeros((nfreq, n_depths, 4), dtype=np.complex128)

    for f in prange(nfreq):
        w = omega[f]

        # --- build full propagator A = P_0 ... P_{n-1} R_half H_half ---
        A = np.eye(4, dtype=np.complex128)
        for i in range(n_finite):
            P = _layer_prop(w, p_si, vp[i], vs[i], rho[i], thickness[i])
            A = _mat4_mul(A, P)

        R_h, kp_z_h, ks_z_h = _build_R(w, p_si, vp[half], vs[half], rho[half])
        H_h = _build_H(kp_z_h, ks_z_h, z_top[half] - z_init)
        A = _mat4_mul(_mat4_mul(A, R_h), H_h)

        # --- solve free-surface BC (Cramer's rule on 2x2) ---
        m00 = A[2, IDX_DS]; m01 = A[2, IDX_DP]
        m10 = A[3, IDX_DS]; m11 = A[3, IDX_DP]
        rhs0 = -A[2, inc_col] * amp
        rhs1 = -A[3, inc_col] * amp
        det = m00 * m11 - m01 * m10
        ds = (rhs0 * m11 - m01 * rhs1) / det
        dp = (m00 * rhs1 - rhs0 * m10) / det

        c = np.zeros(4, dtype=np.complex128)
        c[IDX_DS] = ds
        c[IDX_DP] = dp
        c[inc_col] = amp

        y0 = _mat4_vec(A, c)

        # --- downward-continue to each requested depth ---
        # Pre-compute cumulative inverse propagators at layer boundaries.
        # inv_cum[k] transforms y0 down through layers 0..k-1:
        #   y_at_top_of_layer_k = inv_cum[k] @ y0
        # For k=0 (surface layer) no inverse needed.
        inv_cum = np.eye(4, dtype=np.complex128)  # identity for layer 0

        prev_layer = 0
        for d in range(n_depths):
            li = layer_indices[d]

            # Cascade inverse propagators for layers between prev_layer and li.
            while prev_layer < li:
                Pinv = _inv4x4(
                    _layer_prop(w, p_si, vp[prev_layer], vs[prev_layer],
                                rho[prev_layer], thickness[prev_layer])
                )
                inv_cum = _mat4_mul(Pinv, inv_cum)
                prev_layer += 1

            y_top = _mat4_vec(inv_cum, y0)

            # Final continuation within the target layer.
            z_r = z_depths[d]
            R_d, kp_z_d, ks_z_d = _build_R(w, p_si, vp[li], vs[li], rho[li])
            H_d = _build_H(kp_z_d, ks_z_d, z_r - z_top[li])
            Rinv_d = _inv4x4(R_d)
            M = _mat4_mul(_mat4_mul(R_d, H_d), Rinv_d)
            y_d = _mat4_vec(M, y_top)

            result[f, d, 0] = y_d[0]
            result[f, d, 1] = y_d[1]
            result[f, d, 2] = y_d[2]
            result[f, d, 3] = y_d[3]

    return result


# ------------------------------------------------------------------ #
# Preparation helper (pure Python, runs once)
# ------------------------------------------------------------------ #

def _prepare_depth_args(model: FKLayerModel, z_depths):
    """Sort depths and compute their layer indices.

    Returns ``(z_sorted, layer_indices, sort_order)`` where depths are
    ascending so the kernel can cascade inverse propagators efficiently.
    """
    z_depths = np.atleast_1d(np.asarray(z_depths, dtype=np.float64))
    order = np.argsort(z_depths)
    z_sorted = z_depths[order]
    layer_indices = np.searchsorted(model.z_top, z_sorted, side="right") - 1
    layer_indices = np.clip(layer_indices, 0, model.halfspace_index)
    # Depths at or below the surface might map to -1; clamp.
    layer_indices[z_sorted <= model.z_top[0]] = 0
    return z_sorted, layer_indices.astype(np.int64), order


# ------------------------------------------------------------------ #
# Public wrappers (match haskell.py signatures)
# ------------------------------------------------------------------ #

def surface_state_numba(model: FKLayerModel, omega, p_si, incidence,
                        z_init, amp=None):
    """Drop-in replacement for :func:`fk.propagators.haskell.surface_state` using numba.

    Returns ``y0`` of shape ``(nfreq, 4)``.
    """
    if amp is None:
        half = model.halfspace_index
        v = model.vp[half] if incidence.upper() == "P" else model.vs[half]
        amp = float(np.sin(incidence_angle(p_si, v)))
    inc_col = IDX_UP if incidence.upper() == "P" else IDX_US
    omega = np.asarray(omega, dtype=np.float64)
    z_depths = np.array([0.0], dtype=np.float64)
    layer_indices = np.array([0], dtype=np.int64)
    res = _boundary_states_kernel(
        omega, float(p_si),
        model.vp.astype(np.float64), model.vs.astype(np.float64),
        model.rho.astype(np.float64), model.z_top.astype(np.float64),
        model.thickness.astype(np.float64),
        model.n_finite_layers, float(z_init),
        inc_col, float(amp), z_depths, layer_indices,
    )
    return res[:, 0, :]  # (nfreq, 4)


def states_at_depths_numba(model: FKLayerModel, omega, p_si, incidence,
                           z_init, z_depths, amp=None):
    """Batch-compute state vectors at multiple depths using numba.

    Drop-in replacement for calling :func:`fk.propagators.haskell.state_at_depth` in a loop.

    Returns ``(nfreq, n_depths, 4)`` in the *original* order of ``z_depths``.
    """
    if amp is None:
        half = model.halfspace_index
        v = model.vp[half] if incidence.upper() == "P" else model.vs[half]
        amp = float(np.sin(incidence_angle(p_si, v)))
    inc_col = IDX_UP if incidence.upper() == "P" else IDX_US
    omega = np.asarray(omega, dtype=np.float64)

    z_sorted, layer_idx, order = _prepare_depth_args(model, z_depths)

    res = _boundary_states_kernel(
        omega, float(p_si),
        model.vp.astype(np.float64), model.vs.astype(np.float64),
        model.rho.astype(np.float64), model.z_top.astype(np.float64),
        model.thickness.astype(np.float64),
        model.n_finite_layers, float(z_init),
        inc_col, float(amp), z_sorted, layer_idx,
    )
    # Unsort back to the caller's depth order.
    inv_order = np.argsort(order)
    return res[:, inv_order, :]
