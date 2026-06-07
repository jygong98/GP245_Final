"""2D elastic displacement-based finite-difference solver with C-PML.

Implements the 2nd-order-in-time displacement wave equation on a
collocated grid with high-order spatial FD operators (default 6th order),
following the approach of Liu et al. (2025, SRL / FDFK2D).

Boundary conditions:
- Free-surface at the top (image method / stress-free BC).
- C-PML (convolutional perfectly matched layer) on left, right, bottom.

The core kernels are Numba-accelerated for performance.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .config import HybridConfig


# ===================================================================
# FD coefficient computation (collocated grid)
# ===================================================================

def fd_coef0(norder: int) -> np.ndarray:
    """First-derivative FD coefficients on a staggered (half-grid) grid.

    Returns coef[j, n] for j=1..n, n=1..norder.
    Stencil nodes are at half-integer positions ±(j-½) = ±(2j-1)/2.
    Used for the nested-derivative conservative form (diffx2/diffz2).

    Matches Submain.f90 fd_coef0.
    """
    coef = np.zeros((norder, norder), dtype=np.float64)
    for i in range(1, norder + 1):
        for j in range(1, i + 1):
            tmp1 = 1.0
            tmp2 = 1.0
            tmp3 = 1.0
            for k in range(1, j):
                tmp1 *= (2 * k - 1) * (2 * k - 1)
                tmp2 *= ((2 * j - 1) * (2 * j - 1)
                         - (2 * k - 1) * (2 * k - 1))
            for k in range(j + 1, i + 1):
                tmp1 *= (2 * k - 1) * (2 * k - 1)
                tmp3 *= ((2 * k - 1) * (2 * k - 1)
                         - (2 * j - 1) * (2 * j - 1))
            tmp1 *= (-1) ** (j + 1)
            tmp2 *= (2 * j - 1)
            coef[j - 1, i - 1] = tmp1 / (tmp2 * tmp3)
    return coef


def fd_coef1(norder: int) -> np.ndarray:
    """First-derivative FD coefficients on a collocated grid.

    Returns coef[j, n] for j=1..n, n=1..norder.
    coef[j, n] is the weight for the j-th neighbor pair in an n-th order
    antisymmetric stencil: df/dx ≈ (1/dx) * Σ_j coef[j,n]*(f(x+j) - f(x-j))

    Matches Submain.f90 fd_coef1.
    """
    coef = np.zeros((norder, norder), dtype=np.float64)
    for i in range(1, norder + 1):
        for j in range(1, i + 1):
            tmp1 = 1.0
            tmp2 = 1.0
            tmp3 = 1.0
            for k in range(1, j):
                tmp1 *= k * k
                tmp2 *= (j * j - k * k)
            for k in range(j + 1, i + 1):
                tmp1 *= k * k
                tmp3 *= (k * k - j * j)
            tmp1 *= (-1) ** (j + 1)
            tmp2 *= 2.0 * j
            coef[j - 1, i - 1] = tmp1 / (tmp2 * tmp3)
    return coef


def fd_coef2(norder: int) -> np.ndarray:
    """Second-derivative FD coefficients on a collocated grid.

    Used for stability analysis. Matches Submain.f90 fd_coef2.
    """
    coef = np.zeros((norder, norder), dtype=np.float64)
    for i in range(1, norder + 1):
        for j in range(1, i + 1):
            tmp1 = 1.0
            tmp2 = 1.0
            tmp3 = 1.0
            for k in range(1, j):
                tmp1 *= k * k
                tmp2 *= (j * j - k * k)
            for k in range(j + 1, i + 1):
                tmp1 *= k * k
                tmp3 *= (k * k - j * j)
            tmp1 *= (-1) ** (j + 1)
            tmp2 *= j * j
            coef[j - 1, i - 1] = tmp1 / (tmp2 * tmp3)
    return coef


def build_d2_coef_table(norder: int, coef: np.ndarray,
                        dx: float, dz: float):
    """Second-derivative weights ``2*c_k/(k*h²)`` for each order 1..norder."""
    d2_x = np.zeros((norder, norder), dtype=np.float64)
    d2_z = np.zeros((norder, norder), dtype=np.float64)
    for no in range(1, norder + 1):
        for k in range(1, no + 1):
            ck = coef[k - 1, no - 1]
            d2_x[no - 1, k - 1] = 2.0 * ck / (k * dx * dx)
            d2_z[no - 1, k - 1] = 2.0 * ck / (k * dz * dz)
    return d2_x, d2_z


def check_stability(config: HybridConfig, vp_max: float) -> float:
    """CFL stability check. Returns maximum stable dt.

    Parameters
    ----------
    config : HybridConfig
    vp_max : float
        Maximum P-wave velocity in the model.

    Returns
    -------
    dt_stable : float
        Maximum stable time step.
    """
    norder = config.norder
    coef2 = fd_coef2(norder)
    h = max(config.dx, config.dz)
    cx = (h / config.dx) ** 2
    cz = (h / config.dz) ** 2
    c2_sum = np.sum(np.abs(coef2[:, norder - 1])) + np.sum(coef2[:, norder - 1])
    cfl = np.sqrt(2.0 / ((cx + cz) * c2_sum))
    return cfl * h / vp_max


# ===================================================================
# PML damping profile
# ===================================================================

def pml_damping_profile(npml: int, dx: float, vp_max: float,
                        R0: float = 1e-3, npower: int = 2) -> np.ndarray:
    """Compute PML damping profile d(x) for npml grid points.

    d(x) = -(npower+1) * vp_max * ln(R0) / (2 * L) * (x/L)^npower

    where L = npml * dx is the PML thickness.

    Returns
    -------
    d : ndarray, shape (npml,)
        Damping values from the PML/regular interface inward.
        d[0] ≈ 0 (at interface), d[-1] = max (at outer edge).
    """
    L = npml * dx
    d_max = -(npower + 1) * vp_max * np.log(R0) / (2.0 * L)
    # Normalized distance from interface (0 at interface, 1 at outer edge)
    xi = np.linspace(0.5 / npml, 1.0 - 0.5 / npml, npml)
    return d_max * xi ** npower


def cpml_profiles(npml: int, dx: float, vp_max: float, dt: float,
                   kappa_max: float = 1.0, alpha_max: float = 0.0,
                   npower: int = 2, R0: float = 1e-3):
    """Compute C-PML coefficient arrays for *npml* grid points.

    Returns arrays of shape ``(npml,)`` ordered from the PML/regular
    interface (index 0) to the outer edge (index npml-1).

    Parameters
    ----------
    npml, dx, vp_max, npower, R0 :
        Same as :func:`pml_damping_profile`.
    dt : float
        Time step (s).
    kappa_max : float
        Coordinate stretching factor (>= 1).
    alpha_max : float
        CFS frequency shift (rad/s).

    Returns
    -------
    a, b, d_eff : ndarray, shape (npml,)
        ``b = exp(-(d/kappa + alpha) * dt)``
        ``a = d * (b - 1) / (d + kappa * alpha)`` (zero where denominator vanishes)
        ``d_eff = d / kappa`` (effective damping for the Sochacki-like term)
    """
    d = pml_damping_profile(npml, dx, vp_max, R0=R0, npower=npower)

    xi = np.linspace(0.5 / npml, 1.0 - 0.5 / npml, npml)
    kappa = 1.0 + (kappa_max - 1.0) * xi ** npower
    alpha = alpha_max * (1.0 - xi)

    d_eff = d / kappa

    exponent = -(d / kappa + alpha) * dt
    b = np.exp(exponent)

    denom = d + kappa * alpha
    a = np.where(denom > 0.0, d * (b - 1.0) / denom, 0.0)

    return a.astype(np.float64), b.astype(np.float64), d_eff.astype(np.float64)


# ===================================================================
# Numba kernels for the time-stepping loop
# ===================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _update_displacement(ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                         lam2mu_over_rho, mu_over_rho, lam_mu_over_rho,
                         d2_coef_x, d2_coef_z,
                         coef, norder, dt, dx, dz,
                         iz_start, iz_end, ix_start, ix_end):
    """Advance displacement using 2nd-order time, high-order space FD.

    Variable-order stencil near domain edges (slow path).
    """
    nz_tot = ux_cur.shape[0]
    nx_tot = ux_cur.shape[1]
    dt2 = dt * dt
    idx_idz = 1.0 / dx / dz

    for iz in prange(iz_start, iz_end):
        for ix in range(ix_start, ix_end):
            nox = min(ix, nx_tot - 1 - ix, norder)
            noz = min(iz, nz_tot - 1 - iz, norder)

            if nox < 1 or noz < 1:
                ux_new[iz, ix] = 2.0 * ux_cur[iz, ix] - ux_old[iz, ix]
                uz_new[iz, ix] = 2.0 * uz_cur[iz, ix] - uz_old[iz, ix]
                continue

            d2ux_dx2 = 0.0
            d2uz_dx2 = 0.0
            d2ux_dz2 = 0.0
            d2uz_dz2 = 0.0

            for k in range(nox):
                cx = d2_coef_x[nox - 1, k]
                ux_p = ux_cur[iz, ix + k + 1]
                ux_m = ux_cur[iz, ix - k - 1]
                uz_p = uz_cur[iz, ix + k + 1]
                uz_m = uz_cur[iz, ix - k - 1]
                d2ux_dx2 += cx * (ux_p + ux_m - 2.0 * ux_cur[iz, ix])
                d2uz_dx2 += cx * (uz_p + uz_m - 2.0 * uz_cur[iz, ix])

            for k in range(noz):
                cz = d2_coef_z[noz - 1, k]
                ux_p = ux_cur[iz + k + 1, ix]
                ux_m = ux_cur[iz - k - 1, ix]
                uz_p = uz_cur[iz + k + 1, ix]
                uz_m = uz_cur[iz - k - 1, ix]
                d2ux_dz2 += cz * (ux_p + ux_m - 2.0 * ux_cur[iz, ix])
                d2uz_dz2 += cz * (uz_p + uz_m - 2.0 * uz_cur[iz, ix])

            d2ux_dxdz = 0.0
            d2uz_dxdz = 0.0
            for k in range(1, noz + 1):
                ckz = coef[k - 1, noz - 1]
                dux_x_p = 0.0
                duz_x_p = 0.0
                dux_x_m = 0.0
                duz_x_m = 0.0
                for j in range(1, nox + 1):
                    cjx = coef[j - 1, nox - 1]
                    dux_x_p += cjx * (ux_cur[iz + k, ix + j] - ux_cur[iz + k, ix - j])
                    duz_x_p += cjx * (uz_cur[iz + k, ix + j] - uz_cur[iz + k, ix - j])
                    dux_x_m += cjx * (ux_cur[iz - k, ix + j] - ux_cur[iz - k, ix - j])
                    duz_x_m += cjx * (uz_cur[iz - k, ix + j] - uz_cur[iz - k, ix - j])
                d2ux_dxdz += ckz * (dux_x_p - dux_x_m)
                d2uz_dxdz += ckz * (duz_x_p - duz_x_m)
            d2ux_dxdz *= idx_idz
            d2uz_dxdz *= idx_idz

            l2m = lam2mu_over_rho[iz, ix]
            mu_r = mu_over_rho[iz, ix]
            lm_r = lam_mu_over_rho[iz, ix]

            ax = l2m * d2ux_dx2 + mu_r * d2ux_dz2 + lm_r * d2uz_dxdz
            az = mu_r * d2uz_dx2 + l2m * d2uz_dz2 + lm_r * d2ux_dxdz

            ux_new[iz, ix] = 2.0 * ux_cur[iz, ix] - ux_old[iz, ix] + dt2 * ax
            uz_new[iz, ix] = 2.0 * uz_cur[iz, ix] - uz_old[iz, ix] + dt2 * az


@njit(parallel=True, cache=True, fastmath=True)
def _update_displacement_full(ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                              lam2mu_over_rho, mu_over_rho, lam_mu_over_rho,
                              d2_coef_x, d2_coef_z,
                              coef, norder, dt, dx, dz,
                              iz_start, iz_end, ix_start, ix_end):
    """Interior displacement update with full stencil order.

    Same physics as _update_displacement but assumes all grid points
    are at least *norder* away from array edges.  Eliminates per-point
    min() calls and boundary checks for a significant speedup on the
    interior (which covers >95 percent of grid points for typical grids).

    Caller must guarantee::

        iz_start >= norder  and  iz_end <= nz_tot - norder
        ix_start >= norder  and  ix_end <= nx_tot - norder
    """
    dt2 = dt * dt
    no = norder
    idx_idz = 1.0 / dx / dz

    for iz in prange(iz_start, iz_end):
        for ix in range(ix_start, ix_end):
            d2ux_dx2 = 0.0
            d2uz_dx2 = 0.0
            d2ux_dz2 = 0.0
            d2uz_dz2 = 0.0

            for k in range(no):
                cx = d2_coef_x[no - 1, k]
                cz = d2_coef_z[no - 1, k]
                ux_px = ux_cur[iz, ix + k + 1]
                ux_mx = ux_cur[iz, ix - k - 1]
                uz_px = uz_cur[iz, ix + k + 1]
                uz_mx = uz_cur[iz, ix - k - 1]
                ux_pz = ux_cur[iz + k + 1, ix]
                ux_mz = ux_cur[iz - k - 1, ix]
                uz_pz = uz_cur[iz + k + 1, ix]
                uz_mz = uz_cur[iz - k - 1, ix]
                d2ux_dx2 += cx * (ux_px + ux_mx - 2.0 * ux_cur[iz, ix])
                d2uz_dx2 += cx * (uz_px + uz_mx - 2.0 * uz_cur[iz, ix])
                d2ux_dz2 += cz * (ux_pz + ux_mz - 2.0 * ux_cur[iz, ix])
                d2uz_dz2 += cz * (uz_pz + uz_mz - 2.0 * uz_cur[iz, ix])

            d2ux_dxdz = 0.0
            d2uz_dxdz = 0.0
            for k in range(1, no + 1):
                ckz = coef[k - 1, no - 1]
                dux_x_p = 0.0
                duz_x_p = 0.0
                dux_x_m = 0.0
                duz_x_m = 0.0
                for j in range(1, no + 1):
                    cjx = coef[j - 1, no - 1]
                    dux_x_p += cjx * (ux_cur[iz + k, ix + j] - ux_cur[iz + k, ix - j])
                    duz_x_p += cjx * (uz_cur[iz + k, ix + j] - uz_cur[iz + k, ix - j])
                    dux_x_m += cjx * (ux_cur[iz - k, ix + j] - ux_cur[iz - k, ix - j])
                    duz_x_m += cjx * (uz_cur[iz - k, ix + j] - uz_cur[iz - k, ix - j])
                d2ux_dxdz += ckz * (dux_x_p - dux_x_m)
                d2uz_dxdz += ckz * (duz_x_p - duz_x_m)
            d2ux_dxdz *= idx_idz
            d2uz_dxdz *= idx_idz

            l2m = lam2mu_over_rho[iz, ix]
            mu_r = mu_over_rho[iz, ix]
            lm_r = lam_mu_over_rho[iz, ix]

            ax = l2m * d2ux_dx2 + mu_r * d2ux_dz2 + lm_r * d2uz_dxdz
            az = mu_r * d2uz_dx2 + l2m * d2uz_dz2 + lm_r * d2ux_dxdz

            ux_new[iz, ix] = 2.0 * ux_cur[iz, ix] - ux_old[iz, ix] + dt2 * ax
            uz_new[iz, ix] = 2.0 * uz_cur[iz, ix] - uz_old[iz, ix] + dt2 * az


@njit(parallel=True, cache=True, fastmath=True)
def _update_displacement_conservative(
    ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
    lam2mu, mu, lam, rho,
    coef0, coef1, norder, dt, dx, dz,
    iz_start, iz_end, ix_start, ix_end,
):
    """Conservative-form displacement update with harmonic averaging.

    Implements:
      ax = (1/ρ) * [∂/∂x((λ+2μ)·∂ux/∂x) + ∂/∂z(μ·∂ux/∂z)
                  + ∂/∂x(λ·∂uz/∂z) + ∂/∂z(μ·∂uz/∂x)]
      az = (1/ρ) * [∂/∂x(μ·∂uz/∂x) + ∂/∂z((λ+2μ)·∂uz/∂z)
                  + ∂/∂x(μ·∂ux/∂z) + ∂/∂z(λ·∂ux/∂x)]

    - diffx2/diffz2: conservative nested derivatives with harmonic averaging
      at half-grid points, using staggered coefficients (coef0).
    - diffzx/diffxz: split cross-derivatives with material applied at inner
      derivative position, using collocated coefficients (coef1).
    """
    nz_tot = ux_cur.shape[0]
    nx_tot = ux_cur.shape[1]
    dt2 = dt * dt
    dx2 = dx * dx
    dz2 = dz * dz
    idx = 1.0 / dx
    idz = 1.0 / dz

    for iz in prange(iz_start, iz_end):
        for ix in range(ix_start, ix_end):
            # Available stencil width for outer loop
            nox = min(ix, nx_tot - 1 - ix, norder)
            noz = min(iz, nz_tot - 1 - iz, norder)

            if nox < 1 or noz < 1:
                ux_new[iz, ix] = 2.0 * ux_cur[iz, ix] - ux_old[iz, ix]
                uz_new[iz, ix] = 2.0 * uz_cur[iz, ix] - uz_old[iz, ix]
                continue

            # --- diffx2: ∂/∂x[C · ∂u/∂x] with harmonic averaging ---
            diffx2_l2m_ux = 0.0
            diffx2_mu_uz = 0.0
            for k in range(1, nox + 1):
                # Inner order at shifted positions
                n1 = min(ix + k, nx_tot - 1 - ix - k + 1, norder)
                n2 = min(ix - k + 1, nx_tot - 1 - ix + k, norder)
                if n1 < 1:
                    n1 = 1
                if n2 < 1:
                    n2 = 1

                # Inner staggered derivative at ix+k-½
                du_ux_right = 0.0
                du_uz_right = 0.0
                for j in range(1, n1 + 1):
                    c0j = coef0[j - 1, n1 - 1]
                    du_ux_right += c0j * (ux_cur[iz, ix + j + k - 1]
                                          - ux_cur[iz, ix - j + k])
                    du_uz_right += c0j * (uz_cur[iz, ix + j + k - 1]
                                          - uz_cur[iz, ix - j + k])

                # Inner staggered derivative at ix-k+½
                du_ux_left = 0.0
                du_uz_left = 0.0
                for j in range(1, n2 + 1):
                    c0j = coef0[j - 1, n2 - 1]
                    du_ux_left += c0j * (ux_cur[iz, ix + j - k]
                                         - ux_cur[iz, ix - j - k + 1])
                    du_uz_left += c0j * (uz_cur[iz, ix + j - k]
                                         - uz_cur[iz, ix - j - k + 1])

                # Harmonic mean of (λ+2μ) at half-grid points
                l2m_r = 2.0 / (1.0 / lam2mu[iz, ix + k]
                               + 1.0 / lam2mu[iz, ix + k - 1])
                l2m_l = 2.0 / (1.0 / lam2mu[iz, ix - k + 1]
                               + 1.0 / lam2mu[iz, ix - k])
                # Harmonic mean of μ at half-grid points
                mu_r = 2.0 / (1.0 / mu[iz, ix + k]
                              + 1.0 / mu[iz, ix + k - 1])
                mu_l = 2.0 / (1.0 / mu[iz, ix - k + 1]
                              + 1.0 / mu[iz, ix - k])

                c0k = coef0[k - 1, nox - 1]
                diffx2_l2m_ux += c0k * (l2m_r * du_ux_right
                                        - l2m_l * du_ux_left)
                diffx2_mu_uz += c0k * (mu_r * du_uz_right
                                       - mu_l * du_uz_left)
            diffx2_l2m_ux /= dx2
            diffx2_mu_uz /= dx2

            # --- diffz2: ∂/∂z[C · ∂u/∂z] with harmonic averaging ---
            diffz2_mu_ux = 0.0
            diffz2_l2m_uz = 0.0
            for k in range(1, noz + 1):
                n1 = min(iz + k, nz_tot - 1 - iz - k + 1, norder)
                n2 = min(iz - k + 1, nz_tot - 1 - iz + k, norder)
                if n1 < 1:
                    n1 = 1
                if n2 < 1:
                    n2 = 1

                # Inner staggered derivative at iz+k-½
                du_ux_right = 0.0
                du_uz_right = 0.0
                for j in range(1, n1 + 1):
                    c0j = coef0[j - 1, n1 - 1]
                    du_ux_right += c0j * (ux_cur[iz + j + k - 1, ix]
                                          - ux_cur[iz - j + k, ix])
                    du_uz_right += c0j * (uz_cur[iz + j + k - 1, ix]
                                          - uz_cur[iz - j + k, ix])

                # Inner staggered derivative at iz-k+½
                du_ux_left = 0.0
                du_uz_left = 0.0
                for j in range(1, n2 + 1):
                    c0j = coef0[j - 1, n2 - 1]
                    du_ux_left += c0j * (ux_cur[iz + j - k, ix]
                                         - ux_cur[iz - j - k + 1, ix])
                    du_uz_left += c0j * (uz_cur[iz + j - k, ix]
                                         - uz_cur[iz - j - k + 1, ix])

                # Harmonic mean of μ at half-grid (z-direction)
                mu_top = 2.0 / (1.0 / mu[iz + k, ix]
                                + 1.0 / mu[iz + k - 1, ix])
                mu_bot = 2.0 / (1.0 / mu[iz - k + 1, ix]
                                + 1.0 / mu[iz - k, ix])
                # Harmonic mean of (λ+2μ)
                l2m_top = 2.0 / (1.0 / lam2mu[iz + k, ix]
                                 + 1.0 / lam2mu[iz + k - 1, ix])
                l2m_bot = 2.0 / (1.0 / lam2mu[iz - k + 1, ix]
                                 + 1.0 / lam2mu[iz - k, ix])

                c0k = coef0[k - 1, noz - 1]
                diffz2_mu_ux += c0k * (mu_top * du_ux_right
                                       - mu_bot * du_ux_left)
                diffz2_l2m_uz += c0k * (l2m_top * du_uz_right
                                        - l2m_bot * du_uz_left)
            diffz2_mu_ux /= dz2
            diffz2_l2m_uz /= dz2

            # --- diffzx: ∂/∂x[C · ∂u/∂z] (no harmonic avg) ---
            # Inner z-derivative at shifted x-positions × material,
            # then outer x-derivative
            diffzx_lam_uz = 0.0
            diffzx_mu_ux = 0.0
            for k in range(1, nox + 1):
                noz_r = min(iz, nz_tot - 1 - iz, norder)
                # z-derivative of uz at (iz, ix+k)
                duz_dz_r = 0.0
                dux_dz_r = 0.0
                for j in range(1, noz_r + 1):
                    c1j = coef1[j - 1, noz_r - 1]
                    duz_dz_r += c1j * (uz_cur[iz + j, ix + k]
                                       - uz_cur[iz - j, ix + k])
                    dux_dz_r += c1j * (ux_cur[iz + j, ix + k]
                                       - ux_cur[iz - j, ix + k])
                # z-derivative at (iz, ix-k)
                duz_dz_l = 0.0
                dux_dz_l = 0.0
                for j in range(1, noz_r + 1):
                    c1j = coef1[j - 1, noz_r - 1]
                    duz_dz_l += c1j * (uz_cur[iz + j, ix - k]
                                       - uz_cur[iz - j, ix - k])
                    dux_dz_l += c1j * (ux_cur[iz + j, ix - k]
                                       - ux_cur[iz - j, ix - k])

                c1k = coef1[k - 1, nox - 1]
                # ∂/∂x[λ·∂uz/∂z]: mat applied at inner position
                diffzx_lam_uz += c1k * (lam[iz, ix + k] * duz_dz_r
                                        - lam[iz, ix - k] * duz_dz_l)
                # ∂/∂x[μ·∂ux/∂z]: for az equation
                diffzx_mu_ux += c1k * (mu[iz, ix + k] * dux_dz_r
                                       - mu[iz, ix - k] * dux_dz_l)
            diffzx_lam_uz *= idx * idz
            diffzx_mu_ux *= idx * idz

            # --- diffxz: ∂/∂z[C · ∂u/∂x] (no harmonic avg) ---
            # Inner x-derivative at shifted z-positions × material,
            # then outer z-derivative
            diffxz_mu_uz = 0.0
            diffxz_lam_ux = 0.0
            for k in range(1, noz + 1):
                nox_r = min(ix, nx_tot - 1 - ix, norder)
                # x-derivative of uz at (iz+k, ix)
                duz_dx_t = 0.0
                dux_dx_t = 0.0
                for j in range(1, nox_r + 1):
                    c1j = coef1[j - 1, nox_r - 1]
                    duz_dx_t += c1j * (uz_cur[iz + k, ix + j]
                                       - uz_cur[iz + k, ix - j])
                    dux_dx_t += c1j * (ux_cur[iz + k, ix + j]
                                       - ux_cur[iz + k, ix - j])
                # x-derivative at (iz-k, ix)
                duz_dx_b = 0.0
                dux_dx_b = 0.0
                for j in range(1, nox_r + 1):
                    c1j = coef1[j - 1, nox_r - 1]
                    duz_dx_b += c1j * (uz_cur[iz - k, ix + j]
                                       - uz_cur[iz - k, ix - j])
                    dux_dx_b += c1j * (ux_cur[iz - k, ix + j]
                                       - ux_cur[iz - k, ix - j])

                c1k = coef1[k - 1, noz - 1]
                # ∂/∂z[μ·∂uz/∂x]: for ax equation
                diffxz_mu_uz += c1k * (mu[iz + k, ix] * duz_dx_t
                                       - mu[iz - k, ix] * duz_dx_b)
                # ∂/∂z[λ·∂ux/∂x]: for az equation
                diffxz_lam_ux += c1k * (lam[iz + k, ix] * dux_dx_t
                                        - lam[iz - k, ix] * dux_dx_b)
            diffxz_mu_uz *= idx * idz
            diffxz_lam_ux *= idx * idz

            # Assemble acceleration: a = (1/ρ) * Σ spatial_operators
            inv_rho = 1.0 / rho[iz, ix]
            ax = inv_rho * (diffx2_l2m_ux + diffz2_mu_ux
                            + diffzx_lam_uz + diffxz_mu_uz)
            az = inv_rho * (diffx2_mu_uz + diffz2_l2m_uz
                            + diffzx_mu_ux + diffxz_lam_ux)

            ux_new[iz, ix] = 2.0 * ux_cur[iz, ix] - ux_old[iz, ix] + dt2 * ax
            uz_new[iz, ix] = 2.0 * uz_cur[iz, ix] - uz_old[iz, ix] + dt2 * az


@njit(parallel=True, cache=True, fastmath=True)
def _update_displacement_conservative_full(
    ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
    lam2mu, mu, lam, rho,
    coef0, coef1, norder, dt, dx, dz,
    iz_start, iz_end, ix_start, ix_end,
):
    """Interior conservative update — full stencil order, no boundary checks.

    Caller must guarantee: iz_start >= 2*norder, iz_end <= nz_tot - 2*norder,
    ix_start >= 2*norder, ix_end <= nx_tot - 2*norder.
    """
    dt2 = dt * dt
    dx2 = dx * dx
    dz2 = dz * dz
    idx = 1.0 / dx
    idz = 1.0 / dz
    no = norder

    for iz in prange(iz_start, iz_end):
        for ix in range(ix_start, ix_end):
            # --- diffx2: ∂/∂x[C · ∂u/∂x] ---
            diffx2_l2m_ux = 0.0
            diffx2_mu_uz = 0.0
            for k in range(1, no + 1):
                du_ux_r = 0.0
                du_uz_r = 0.0
                du_ux_l = 0.0
                du_uz_l = 0.0
                for j in range(1, no + 1):
                    c0j = coef0[j - 1, no - 1]
                    du_ux_r += c0j * (ux_cur[iz, ix + j + k - 1]
                                      - ux_cur[iz, ix - j + k])
                    du_uz_r += c0j * (uz_cur[iz, ix + j + k - 1]
                                      - uz_cur[iz, ix - j + k])
                    du_ux_l += c0j * (ux_cur[iz, ix + j - k]
                                      - ux_cur[iz, ix - j - k + 1])
                    du_uz_l += c0j * (uz_cur[iz, ix + j - k]
                                      - uz_cur[iz, ix - j - k + 1])

                l2m_r = 2.0 / (1.0 / lam2mu[iz, ix + k]
                               + 1.0 / lam2mu[iz, ix + k - 1])
                l2m_l = 2.0 / (1.0 / lam2mu[iz, ix - k + 1]
                               + 1.0 / lam2mu[iz, ix - k])
                mu_r = 2.0 / (1.0 / mu[iz, ix + k]
                              + 1.0 / mu[iz, ix + k - 1])
                mu_l = 2.0 / (1.0 / mu[iz, ix - k + 1]
                              + 1.0 / mu[iz, ix - k])

                c0k = coef0[k - 1, no - 1]
                diffx2_l2m_ux += c0k * (l2m_r * du_ux_r - l2m_l * du_ux_l)
                diffx2_mu_uz += c0k * (mu_r * du_uz_r - mu_l * du_uz_l)
            diffx2_l2m_ux /= dx2
            diffx2_mu_uz /= dx2

            # --- diffz2: ∂/∂z[C · ∂u/∂z] ---
            diffz2_mu_ux = 0.0
            diffz2_l2m_uz = 0.0
            for k in range(1, no + 1):
                du_ux_r = 0.0
                du_uz_r = 0.0
                du_ux_l = 0.0
                du_uz_l = 0.0
                for j in range(1, no + 1):
                    c0j = coef0[j - 1, no - 1]
                    du_ux_r += c0j * (ux_cur[iz + j + k - 1, ix]
                                      - ux_cur[iz - j + k, ix])
                    du_uz_r += c0j * (uz_cur[iz + j + k - 1, ix]
                                      - uz_cur[iz - j + k, ix])
                    du_ux_l += c0j * (ux_cur[iz + j - k, ix]
                                      - ux_cur[iz - j - k + 1, ix])
                    du_uz_l += c0j * (uz_cur[iz + j - k, ix]
                                      - uz_cur[iz - j - k + 1, ix])

                mu_t = 2.0 / (1.0 / mu[iz + k, ix]
                              + 1.0 / mu[iz + k - 1, ix])
                mu_b = 2.0 / (1.0 / mu[iz - k + 1, ix]
                              + 1.0 / mu[iz - k, ix])
                l2m_t = 2.0 / (1.0 / lam2mu[iz + k, ix]
                               + 1.0 / lam2mu[iz + k - 1, ix])
                l2m_b = 2.0 / (1.0 / lam2mu[iz - k + 1, ix]
                               + 1.0 / lam2mu[iz - k, ix])

                c0k = coef0[k - 1, no - 1]
                diffz2_mu_ux += c0k * (mu_t * du_ux_r - mu_b * du_ux_l)
                diffz2_l2m_uz += c0k * (l2m_t * du_uz_r - l2m_b * du_uz_l)
            diffz2_mu_ux /= dz2
            diffz2_l2m_uz /= dz2

            # --- diffzx: ∂/∂x[C · ∂u/∂z] ---
            diffzx_lam_uz = 0.0
            diffzx_mu_ux = 0.0
            for k in range(1, no + 1):
                duz_dz_r = 0.0
                dux_dz_r = 0.0
                duz_dz_l = 0.0
                dux_dz_l = 0.0
                for j in range(1, no + 1):
                    c1j = coef1[j - 1, no - 1]
                    duz_dz_r += c1j * (uz_cur[iz + j, ix + k]
                                       - uz_cur[iz - j, ix + k])
                    dux_dz_r += c1j * (ux_cur[iz + j, ix + k]
                                       - ux_cur[iz - j, ix + k])
                    duz_dz_l += c1j * (uz_cur[iz + j, ix - k]
                                       - uz_cur[iz - j, ix - k])
                    dux_dz_l += c1j * (ux_cur[iz + j, ix - k]
                                       - ux_cur[iz - j, ix - k])

                c1k = coef1[k - 1, no - 1]
                diffzx_lam_uz += c1k * (lam[iz, ix + k] * duz_dz_r
                                        - lam[iz, ix - k] * duz_dz_l)
                diffzx_mu_ux += c1k * (mu[iz, ix + k] * dux_dz_r
                                       - mu[iz, ix - k] * dux_dz_l)
            diffzx_lam_uz *= idx * idz
            diffzx_mu_ux *= idx * idz

            # --- diffxz: ∂/∂z[C · ∂u/∂x] ---
            diffxz_mu_uz = 0.0
            diffxz_lam_ux = 0.0
            for k in range(1, no + 1):
                duz_dx_t = 0.0
                dux_dx_t = 0.0
                duz_dx_b = 0.0
                dux_dx_b = 0.0
                for j in range(1, no + 1):
                    c1j = coef1[j - 1, no - 1]
                    duz_dx_t += c1j * (uz_cur[iz + k, ix + j]
                                       - uz_cur[iz + k, ix - j])
                    dux_dx_t += c1j * (ux_cur[iz + k, ix + j]
                                       - ux_cur[iz + k, ix - j])
                    duz_dx_b += c1j * (uz_cur[iz - k, ix + j]
                                       - uz_cur[iz - k, ix - j])
                    dux_dx_b += c1j * (ux_cur[iz - k, ix + j]
                                       - ux_cur[iz - k, ix - j])

                c1k = coef1[k - 1, no - 1]
                diffxz_mu_uz += c1k * (mu[iz + k, ix] * duz_dx_t
                                       - mu[iz - k, ix] * duz_dx_b)
                diffxz_lam_ux += c1k * (lam[iz + k, ix] * dux_dx_t
                                        - lam[iz - k, ix] * dux_dx_b)
            diffxz_mu_uz *= idx * idz
            diffxz_lam_ux *= idx * idz

            # Assemble
            inv_rho = 1.0 / rho[iz, ix]
            ax = inv_rho * (diffx2_l2m_ux + diffz2_mu_ux
                            + diffzx_lam_uz + diffxz_mu_uz)
            az = inv_rho * (diffx2_mu_uz + diffz2_l2m_uz
                            + diffzx_mu_ux + diffxz_lam_ux)

            ux_new[iz, ix] = 2.0 * ux_cur[iz, ix] - ux_old[iz, ix] + dt2 * ax
            uz_new[iz, ix] = 2.0 * uz_cur[iz, ix] - uz_old[iz, ix] + dt2 * az


@njit(parallel=True, cache=True, fastmath=True)
def _apply_pml_damping(ux_new, uz_new, ux_cur, ux_old, uz_cur, uz_old,
                       dt, dx_arr, dz_arr, nz_tot, nx_tot, npml, norder):
    """Apply C-PML damping to displacement in the absorbing layers.

    Only iterates over PML strips (left, right, bottom) rather than
    the full grid, reducing iteration count by ~10-20x for typical grids.

    Uses the damped wave equation formulation (Sochacki et al., 1987):
      u_new_damped = (u_new_undamped + (d*dt/2)*u_old) / (1 + d*dt/2)
    """
    nz = nz_tot - npml
    nx_int = nx_tot - 2 * npml
    ix_right = npml + nx_int

    # Left and right PML strips (all rows, handles corners)
    for iz in prange(nz_tot):
        dz_val = dz_arr[iz]
        for ix in range(npml):
            d_total = dx_arr[ix] + dz_val
            alpha = d_total * dt * 0.5
            denom = 1.0 + alpha
            ux_new[iz, ix] = (ux_new[iz, ix] + alpha * ux_old[iz, ix]) / denom
            uz_new[iz, ix] = (uz_new[iz, ix] + alpha * uz_old[iz, ix]) / denom
        for ix in range(ix_right, nx_tot):
            d_total = dx_arr[ix] + dz_val
            alpha = d_total * dt * 0.5
            denom = 1.0 + alpha
            ux_new[iz, ix] = (ux_new[iz, ix] + alpha * ux_old[iz, ix]) / denom
            uz_new[iz, ix] = (uz_new[iz, ix] + alpha * uz_old[iz, ix]) / denom

    # Bottom PML interior columns (corners already handled above)
    for iz in prange(nz, nz_tot):
        dz_val = dz_arr[iz]
        for ix in range(npml, ix_right):
            alpha = dz_val * dt * 0.5
            denom = 1.0 + alpha
            ux_new[iz, ix] = (ux_new[iz, ix] + alpha * ux_old[iz, ix]) / denom
            uz_new[iz, ix] = (uz_new[iz, ix] + alpha * uz_old[iz, ix]) / denom


@njit(parallel=True, cache=True, fastmath=True)
def _apply_cpml_damping(ux_new, uz_new, ux_cur, ux_old, uz_cur, uz_old,
                        dt, dx_deff, dz_deff, dx_a, dz_a, dx_b, dz_b,
                        psi_ux, psi_uz,
                        nz_tot, nx_tot, npml, nz, nx):
    """Apply C-PML damping with CFS memory variables.

    Replaces the Sochacki-only ``_apply_pml_damping`` with CFS-PML:
    - Uses ``d_eff = d / kappa`` for improved angular absorption.
    - Maintains memory variables ``psi`` for the CFS frequency shift,
      which greatly improves absorption of low-frequency and
      grazing-incidence waves.

    Update rule at each PML point::

        v = (u_cur - u_old) / dt          # velocity approximation
        psi = b * psi_prev + a * v        # memory variable update
        alpha = d_eff * dt / 2
        u_new = (u_new_undamped + alpha * u_old + dt * psi) / (1 + alpha)
    """
    ix_right = npml + nx

    # Left and right PML strips (all rows, handles corners)
    for iz in prange(nz_tot):
        dz_d = dz_deff[iz]
        dz_a_val = dz_a[iz]
        dz_b_val = dz_b[iz]
        for ix in range(npml):
            d_total = dx_deff[ix] + dz_d
            alpha = d_total * dt * 0.5
            denom = 1.0 + alpha

            # CFS memory variable update (combined x + z damping)
            a_total = dx_a[ix] + dz_a_val
            b_total_ux = dx_b[ix] * dz_b_val if (dx_b[ix] < 1.0 and dz_b_val < 1.0) else max(dx_b[ix], dz_b_val)
            # Use simpler additive approach for combined damping
            b_val = min(dx_b[ix], dz_b_val) if (dx_b[ix] < 1.0 or dz_b_val < 1.0) else 1.0

            vx = (ux_cur[iz, ix] - ux_old[iz, ix]) / dt
            vz = (uz_cur[iz, ix] - uz_old[iz, ix]) / dt

            psi_ux[iz, ix] = b_val * psi_ux[iz, ix] + a_total * vx
            psi_uz[iz, ix] = b_val * psi_uz[iz, ix] + a_total * vz

            ux_new[iz, ix] = (ux_new[iz, ix] + alpha * ux_old[iz, ix] + dt * psi_ux[iz, ix]) / denom
            uz_new[iz, ix] = (uz_new[iz, ix] + alpha * uz_old[iz, ix] + dt * psi_uz[iz, ix]) / denom

        for ix in range(ix_right, nx_tot):
            d_total = dx_deff[ix] + dz_d
            alpha = d_total * dt * 0.5
            denom = 1.0 + alpha

            a_total = dx_a[ix] + dz_a_val
            b_val = min(dx_b[ix], dz_b_val) if (dx_b[ix] < 1.0 or dz_b_val < 1.0) else 1.0

            vx = (ux_cur[iz, ix] - ux_old[iz, ix]) / dt
            vz = (uz_cur[iz, ix] - uz_old[iz, ix]) / dt

            psi_ux[iz, ix] = b_val * psi_ux[iz, ix] + a_total * vx
            psi_uz[iz, ix] = b_val * psi_uz[iz, ix] + a_total * vz

            ux_new[iz, ix] = (ux_new[iz, ix] + alpha * ux_old[iz, ix] + dt * psi_ux[iz, ix]) / denom
            uz_new[iz, ix] = (uz_new[iz, ix] + alpha * uz_old[iz, ix] + dt * psi_uz[iz, ix]) / denom

    # Bottom PML interior columns (corners already handled above)
    for iz in prange(nz, nz_tot):
        dz_d = dz_deff[iz]
        dz_a_val = dz_a[iz]
        dz_b_val = dz_b[iz]
        for ix in range(npml, ix_right):
            alpha = dz_d * dt * 0.5
            denom = 1.0 + alpha

            vx = (ux_cur[iz, ix] - ux_old[iz, ix]) / dt
            vz = (uz_cur[iz, ix] - uz_old[iz, ix]) / dt

            psi_ux[iz, ix] = dz_b_val * psi_ux[iz, ix] + dz_a_val * vx
            psi_uz[iz, ix] = dz_b_val * psi_uz[iz, ix] + dz_a_val * vz

            ux_new[iz, ix] = (ux_new[iz, ix] + alpha * ux_old[iz, ix] + dt * psi_ux[iz, ix]) / denom
            uz_new[iz, ix] = (uz_new[iz, ix] + alpha * uz_old[iz, ix] + dt * psi_uz[iz, ix]) / denom


@njit(cache=True, fastmath=True)
def _apply_free_surface(ux, uz, lam, mu, norder, coef, dz, iz_surf, nx_tot,
                        dx=0.0):
    """Apply stress-free boundary condition at iz_surf (top surface).

    σ_zz = 0 at z=0: λ*∂ux/∂x + (λ+2μ)*∂uz/∂z = 0
    σ_xz = 0 at z=0: μ*(∂ux/∂z + ∂uz/∂x) = 0

    When dx > 0, the proper stress-free conditions are used:
      ux[0] = ux[1] + dz * ∂uz/∂x  (from σ_xz = 0)
      uz[0] = uz[1] + dz * λ/(λ+2μ) * ∂ux/∂x  (from σ_zz = 0)
    where x-derivatives are evaluated at iz=1 (first interior row).

    When dx == 0, falls back to simple copy (ux[0]=ux[1], uz[0]=uz[1]).
    """
    nz_tot = ux.shape[0]
    if iz_surf != 0 or nz_tot <= 2:
        return

    if dx > 0.0:
        idx = 1.0 / dx
        for ix in range(nx_tot):
            nox = min(ix, nx_tot - 1 - ix, norder)
            if nox < 1:
                ux[0, ix] = ux[1, ix]
                uz[0, ix] = uz[1, ix]
                continue

            # x-derivatives at iz=1 (first fully-updated interior row),
            # used as approximation for derivatives at the surface.
            dux_dx = 0.0
            duz_dx = 0.0
            for k in range(1, nox + 1):
                ck = coef[k - 1, nox - 1]
                dux_dx += ck * (ux[1, ix + k] - ux[1, ix - k])
                duz_dx += ck * (uz[1, ix + k] - uz[1, ix - k])
            dux_dx *= idx
            duz_dx *= idx

            lam_ix = lam[0, ix]
            mu_ix = mu[0, ix]
            lam2mu = lam_ix + 2.0 * mu_ix
            ratio = lam_ix / lam2mu

            # 2nd-order one-sided extrapolation with stress-free conditions.
            # σ_xz=0 → ∂ux/∂z|_0 = -∂uz/∂x|_0
            # σ_zz=0 → ∂uz/∂z|_0 = -(λ/(λ+2μ)) * ∂ux/∂x|_0
            # Using (-3f[0]+4f[1]-f[2])/(2dz) = f'(0):
            #   f[0] = (4f[1] - f[2] + 2dz*f'(0)) / 3
            ux[0, ix] = (4.0 * ux[1, ix] - ux[2, ix]
                         + 2.0 * dz * duz_dx) / 3.0
            uz[0, ix] = (4.0 * uz[1, ix] - uz[2, ix]
                         + 2.0 * dz * ratio * dux_dx) / 3.0
    else:
        for ix in range(nx_tot):
            ux[0, ix] = ux[1, ix]
            uz[0, ix] = uz[1, ix]


# ===================================================================
# Split FD update helpers (regular interior + PML strips)
# ===================================================================

def fd_update_regular(solver: "FDDisplacementSolver"):
    """FD update for regular-domain nodes (boundary rows + interior)."""
    npml = solver.npml
    norder = solver.norder
    iz_start = solver.iz_start
    nz = solver.nz

    if solver.conservative:
        # Conservative form: wider stencil (2*norder), single kernel handles
        # both boundary and interior
        _update_displacement_conservative(
            solver.ux_new, solver.uz_new,
            solver.ux_cur, solver.uz_cur,
            solver.ux_old, solver.uz_old,
            solver.lam2mu, solver.mu_raw, solver.lam_raw, solver.rho,
            solver.coef0, solver.coef, norder, solver.dt, solver.dx, solver.dz,
            iz_start, nz, npml, npml + solver.nx,
        )
    else:
        iz_bnd = min(norder, nz)
        if iz_start < iz_bnd:
            _update_displacement(
                solver.ux_new, solver.uz_new,
                solver.ux_cur, solver.uz_cur,
                solver.ux_old, solver.uz_old,
                solver.lam2mu_over_rho, solver.mu_over_rho,
                solver.lam_mu_over_rho,
                solver.d2_coef_x, solver.d2_coef_z,
                solver.coef, norder, solver.dt, solver.dx, solver.dz,
                iz_start, iz_bnd, npml, npml + solver.nx,
            )
        iz_int = max(iz_start, norder)
        if iz_int < nz:
            _update_displacement_full(
                solver.ux_new, solver.uz_new,
                solver.ux_cur, solver.uz_cur,
                solver.ux_old, solver.uz_old,
                solver.lam2mu_over_rho, solver.mu_over_rho,
                solver.lam_mu_over_rho,
                solver.d2_coef_x, solver.d2_coef_z,
                solver.coef, norder, solver.dt, solver.dx, solver.dz,
                iz_int, nz, npml, npml + solver.nx,
            )


def fd_update_pml(solver: "FDDisplacementSolver"):
    """FD update for PML-domain nodes (left, right, bottom strips)."""
    npml = solver.npml
    nz, nx = solver.nz, solver.nx
    nx_tot = solver.nx_tot
    nz_tot = solver.nz_tot

    if solver.conservative:
        if npml > 1:
            _update_displacement_conservative(
                solver.ux_new, solver.uz_new,
                solver.ux_cur, solver.uz_cur,
                solver.ux_old, solver.uz_old,
                solver.lam2mu, solver.mu_raw, solver.lam_raw, solver.rho,
                solver.coef0, solver.coef, solver.norder,
                solver.dt, solver.dx, solver.dz,
                1, nz_tot, 1, npml,
            )
        if npml + nx < nx_tot:
            _update_displacement_conservative(
                solver.ux_new, solver.uz_new,
                solver.ux_cur, solver.uz_cur,
                solver.ux_old, solver.uz_old,
                solver.lam2mu, solver.mu_raw, solver.lam_raw, solver.rho,
                solver.coef0, solver.coef, solver.norder,
                solver.dt, solver.dx, solver.dz,
                1, nz_tot, npml + nx, nx_tot - 1,
            )
        if nz < nz_tot:
            _update_displacement_conservative(
                solver.ux_new, solver.uz_new,
                solver.ux_cur, solver.uz_cur,
                solver.ux_old, solver.uz_old,
                solver.lam2mu, solver.mu_raw, solver.lam_raw, solver.rho,
                solver.coef0, solver.coef, solver.norder,
                solver.dt, solver.dx, solver.dz,
                nz, nz_tot, npml, npml + nx,
            )
    else:
        if npml > 1:
            _update_displacement(
                solver.ux_new, solver.uz_new,
                solver.ux_cur, solver.uz_cur,
                solver.ux_old, solver.uz_old,
                solver.lam2mu_over_rho, solver.mu_over_rho,
                solver.lam_mu_over_rho,
                solver.d2_coef_x, solver.d2_coef_z,
                solver.coef, solver.norder, solver.dt, solver.dx, solver.dz,
                1, nz_tot, 1, npml,
            )
        if npml + nx < nx_tot:
            _update_displacement(
                solver.ux_new, solver.uz_new,
                solver.ux_cur, solver.uz_cur,
                solver.ux_old, solver.uz_old,
                solver.lam2mu_over_rho, solver.mu_over_rho,
                solver.lam_mu_over_rho,
                solver.d2_coef_x, solver.d2_coef_z,
                solver.coef, solver.norder, solver.dt, solver.dx, solver.dz,
                1, nz_tot, npml + nx, nx_tot - 1,
            )
        if nz < nz_tot:
            _update_displacement(
                solver.ux_new, solver.uz_new,
                solver.ux_cur, solver.uz_cur,
                solver.ux_old, solver.uz_old,
                solver.lam2mu_over_rho, solver.mu_over_rho,
                solver.lam_mu_over_rho,
                solver.d2_coef_x, solver.d2_coef_z,
                solver.coef, solver.norder, solver.dt, solver.dx, solver.dz,
                nz, nz_tot, npml, npml + nx,
            )


def fd_update_split(solver: "FDDisplacementSolver"):
    """Full-domain FD update using split regular + PML kernel strategy."""
    fd_update_regular(solver)
    fd_update_pml(solver)


# ===================================================================
# High-level FD solver class
# ===================================================================

class FDDisplacementSolver:
    """2D elastic displacement FD solver with PML and free-surface BC.

    Parameters
    ----------
    config : HybridConfig
        Simulation grid parameters.
    rho, lam, mu : ndarray, shape (nz_total, nx_total)
        Material properties on the extended grid (including PML).
        nz_total = nz + npml (top is free-surface, no PML there).
        nx_total = nx + 2*npml.
    conservative : bool, optional
        If True (default), use conservative-form FD with harmonic averaging
        at material discontinuities, matching Fortran FDFK2D. If False, use
        the legacy non-conservative form C·∂²u/∂x².
    """

    def __init__(self, config: HybridConfig,
                 rho: np.ndarray, lam: np.ndarray, mu: np.ndarray,
                 conservative: bool = True):
        self.config = config
        self.norder = config.norder
        self.npml = config.npml
        self.nx = config.nx
        self.nz = config.nz
        self.dx = config.dx
        self.dz = config.dz
        self.dt = config.dt
        self.conservative = conservative

        # Extended grid dimensions: PML on left, right, bottom only
        # Top has free surface (no PML)
        self.nx_tot = config.nx + 2 * config.npml
        self.nz_tot = config.nz + config.npml

        # Material arrays (extended grid)
        assert rho.shape == (self.nz_tot, self.nx_tot)
        self.rho = rho.astype(np.float64)
        self.lam = lam.astype(np.float64)
        self.mu = mu.astype(np.float64)

        # FD coefficients
        self.coef = fd_coef1(config.norder)

        # Conservative-form specific arrays
        if conservative:
            self.coef0 = fd_coef0(config.norder)
            self.lam2mu = (self.lam + 2.0 * self.mu).copy()
            self.mu_raw = self.mu.copy()
            self.lam_raw = self.lam.copy()

        # Precomputed elastic combination quantities (legacy non-conservative)
        self.lam2mu_over_rho = (self.lam + 2.0 * self.mu) / self.rho
        self.mu_over_rho = self.mu / self.rho
        self.lam_mu_over_rho = (self.lam + self.mu) / self.rho

        # Precomputed second-derivative coefficients (legacy non-conservative)
        self.d2_coef_x, self.d2_coef_z = build_d2_coef_table(
            config.norder, self.coef, config.dx, config.dz,
        )

        # PML damping profiles
        vp_max = float(np.max(np.sqrt((lam + 2.0 * mu) / rho)))
        self.dx_damp = np.zeros(self.nx_tot, dtype=np.float64)
        self.dz_damp = np.zeros(self.nz_tot, dtype=np.float64)
        d_profile_x = pml_damping_profile(config.npml, config.dx, vp_max)
        d_profile_z = pml_damping_profile(config.npml, config.dz, vp_max)

        # Left PML: indices [0, npml)
        self.dx_damp[:config.npml] = d_profile_x[::-1]
        # Right PML: indices [npml+nx, npml+nx+npml)
        self.dx_damp[config.npml + config.nx:] = d_profile_x
        # Bottom PML: indices [nz, nz+npml)
        self.dz_damp[config.nz:] = d_profile_z

        # C-PML profiles (used when kappa_max > 1 or alpha_max > 0)
        self.use_cpml = config.kappa_max > 1.0 or config.alpha_max > 0.0
        if self.use_cpml:
            ax, bx, deff_x = cpml_profiles(
                config.npml, config.dx, vp_max, config.dt,
                kappa_max=config.kappa_max, alpha_max=config.alpha_max,
                npower=config.cpml_npower,
            )
            az, bz, deff_z = cpml_profiles(
                config.npml, config.dz, vp_max, config.dt,
                kappa_max=config.kappa_max, alpha_max=config.alpha_max,
                npower=config.cpml_npower,
            )

            self.dx_deff = np.zeros(self.nx_tot, dtype=np.float64)
            self.dx_a = np.zeros(self.nx_tot, dtype=np.float64)
            self.dx_b = np.ones(self.nx_tot, dtype=np.float64)
            self.dx_deff[:config.npml] = deff_x[::-1]
            self.dx_deff[config.npml + config.nx:] = deff_x
            self.dx_a[:config.npml] = ax[::-1]
            self.dx_a[config.npml + config.nx:] = ax
            self.dx_b[:config.npml] = bx[::-1]
            self.dx_b[config.npml + config.nx:] = bx

            self.dz_deff = np.zeros(self.nz_tot, dtype=np.float64)
            self.dz_a = np.zeros(self.nz_tot, dtype=np.float64)
            self.dz_b = np.ones(self.nz_tot, dtype=np.float64)
            self.dz_deff[config.nz:] = deff_z
            self.dz_a[config.nz:] = az
            self.dz_b[config.nz:] = bz

            # C-PML memory variables (full-grid, only updated in PML strips)
            self.psi_ux = np.zeros((self.nz_tot, self.nx_tot), dtype=np.float64)
            self.psi_uz = np.zeros((self.nz_tot, self.nx_tot), dtype=np.float64)

        # Displacement fields: (nz_tot, nx_tot) - three time levels
        self.ux_cur = np.zeros((self.nz_tot, self.nx_tot), dtype=np.float64)
        self.uz_cur = np.zeros((self.nz_tot, self.nx_tot), dtype=np.float64)
        self.ux_old = np.zeros((self.nz_tot, self.nx_tot), dtype=np.float64)
        self.uz_old = np.zeros((self.nz_tot, self.nx_tot), dtype=np.float64)
        self.ux_new = np.zeros((self.nz_tot, self.nx_tot), dtype=np.float64)
        self.uz_new = np.zeros((self.nz_tot, self.nx_tot), dtype=np.float64)

        # Surface index in the extended grid
        self.iz_surf = 0

        # Interior update bounds (regular + PML, but not the outermost edge)
        self.iz_start = 1  # skip surface row
        self.iz_end = self.nz_tot - 1
        self.ix_start = 1
        self.ix_end = self.nx_tot - 1

        # Check stability
        dt_stable = check_stability(config, vp_max)
        if config.dt > dt_stable:
            raise ValueError(
                f"dt={config.dt:.6f}s exceeds stable limit {dt_stable:.6f}s. "
                f"Reduce dt or coarsen the grid."
            )

    def reset(self):
        """Reset displacement fields and C-PML memory variables to zero."""
        self.ux_cur[:] = 0.0
        self.uz_cur[:] = 0.0
        self.ux_old[:] = 0.0
        self.uz_old[:] = 0.0
        self.ux_new[:] = 0.0
        self.uz_new[:] = 0.0
        if self.use_cpml:
            self.psi_ux[:] = 0.0
            self.psi_uz[:] = 0.0

    def step(self):
        """Advance one time step (backward-compatible, no source injection)."""
        self.compute_new()
        self.apply_boundary()
        self.rotate()

    def compute_new(self):
        """Compute u_new from spatial operators using split kernel strategy."""
        fd_update_split(self)

    def apply_boundary(self):
        """Apply PML damping and free-surface BC to u_new."""
        if self.use_cpml:
            _apply_cpml_damping(
                self.ux_new, self.uz_new,
                self.ux_cur, self.ux_old,
                self.uz_cur, self.uz_old,
                self.dt, self.dx_deff, self.dz_deff,
                self.dx_a, self.dz_a, self.dx_b, self.dz_b,
                self.psi_ux, self.psi_uz,
                self.nz_tot, self.nx_tot, self.npml, self.nz, self.nx,
            )
        else:
            _apply_pml_damping(
                self.ux_new, self.uz_new,
                self.ux_cur, self.ux_old,
                self.uz_cur, self.uz_old,
                self.dt, self.dx_damp, self.dz_damp,
                self.nz_tot, self.nx_tot, self.npml, self.norder,
            )

        _apply_free_surface(
            self.ux_new, self.uz_new,
            self.lam, self.mu, self.norder, self.coef,
            self.dz, self.iz_surf, self.nx_tot, self.dx,
        )

    def inject_source(self, source, it: int):
        """Inject source into u_new — delegates to :func:`sources.inject_source`."""
        from .sources import inject_source as _inject
        _inject(self, source, it)

    def rotate(self):
        """Rotate time levels: old <- cur <- new <- old."""
        self.ux_old, self.ux_cur, self.ux_new = (
            self.ux_cur, self.ux_new, self.ux_old
        )
        self.uz_old, self.uz_cur, self.uz_new = (
            self.uz_cur, self.uz_new, self.uz_old
        )

    def get_displacement(self, component: str = "both"):
        """Get current displacement in the regular domain (no PML padding).

        Returns arrays of shape (nz, nx).
        """
        npml = self.npml
        ux_int = self.ux_cur[:self.nz, npml:npml + self.nx]
        uz_int = self.uz_cur[:self.nz, npml:npml + self.nx]
        if component == "x":
            return ux_int
        elif component == "z":
            return uz_int
        return ux_int, uz_int

    def to_extended_index(self, ix_reg: int, iz_reg: int) -> tuple[int, int]:
        """Convert regular-domain indices to extended-grid indices."""
        return iz_reg, ix_reg + self.npml

    @property
    def ux_extended(self) -> np.ndarray:
        """Current ux on the full extended grid (including PML)."""
        return self.ux_cur

    @property
    def uz_extended(self) -> np.ndarray:
        """Current uz on the full extended grid (including PML)."""
        return self.uz_cur


def build_material_arrays(config: HybridConfig,
                          vp_2d: np.ndarray,
                          vs_2d: np.ndarray,
                          rho_2d: np.ndarray):
    """Build extended material arrays (with PML padding) from interior model.

    Parameters
    ----------
    config : HybridConfig
    vp_2d, vs_2d, rho_2d : ndarray, shape (nz, nx)
        P-velocity, S-velocity, density on the interior grid.

    Returns
    -------
    rho, lam, mu : ndarray, shape (nz+npml, nx+2*npml)
    """
    npml = config.npml
    nx, nz = config.nx, config.nz
    nz_tot = nz + npml
    nx_tot = nx + 2 * npml

    rho_ext = np.zeros((nz_tot, nx_tot), dtype=np.float64)
    vp_ext = np.zeros((nz_tot, nx_tot), dtype=np.float64)
    vs_ext = np.zeros((nz_tot, nx_tot), dtype=np.float64)

    # Fill interior
    rho_ext[:nz, npml:npml + nx] = rho_2d
    vp_ext[:nz, npml:npml + nx] = vp_2d
    vs_ext[:nz, npml:npml + nx] = vs_2d

    # Extend into left PML (copy leftmost column)
    rho_ext[:nz, :npml] = rho_2d[:, 0:1]
    vp_ext[:nz, :npml] = vp_2d[:, 0:1]
    vs_ext[:nz, :npml] = vs_2d[:, 0:1]

    # Extend into right PML
    rho_ext[:nz, npml + nx:] = rho_2d[:, -1:]
    vp_ext[:nz, npml + nx:] = vp_2d[:, -1:]
    vs_ext[:nz, npml + nx:] = vs_2d[:, -1:]

    # Extend into bottom PML (copy bottom row of each column)
    rho_ext[nz:, :] = rho_ext[nz - 1:nz, :]
    vp_ext[nz:, :] = vp_ext[nz - 1:nz, :]
    vs_ext[nz:, :] = vs_ext[nz - 1:nz, :]

    mu_ext = rho_ext * vs_ext ** 2
    lam_ext = rho_ext * (vp_ext ** 2 - 2.0 * vs_ext ** 2)

    return rho_ext, lam_ext, mu_ext


# ===================================================================
# Re-exports for backward compatibility — canonical definitions live
# in hybrid.sources
# ===================================================================
from .sources import (  # noqa: E402, F401
    PointBodyForceSource,
    LineBodyForceSource,
    ExplosiveLineSource,
    ExplosivePointSource,
    run_standalone,
)

