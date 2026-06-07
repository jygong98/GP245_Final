"""Top-level hybrid FD-FK engine.

Orchestrates the full simulation:
1. Pre-compute FK background wavefield at coupling-zone boundaries.
2. Set up the 2D FD displacement solver.
3. Time-step with wavefield injection.
4. Extract seismograms at receiver positions.

Wavefield injection uses total-field / scattered-field (TF/SF) two-pass
coupling-zone injection (``hybrid.injection``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numba import njit

from fk.model import FKLayerModel
from fk.source import RickerSource
from fk.geometry import incidence_angle

from .config import HybridConfig
from .background import (
    compute_boundary_wavefield, BoundaryWavefield,
    compute_fk_greens, boundary_wavefield_from_greens, FKGreensCache,
    synthesis_length,
)
from .fd_displacement import (
    FDDisplacementSolver,
    build_material_arrays,
    fd_update_regular,
    fd_update_pml,
    _update_displacement,
    _update_displacement_full,
    _update_displacement_conservative,
    _apply_pml_damping,
    _apply_cpml_damping,
    _apply_free_surface,
)
from .injection import (
    inject_for_regular_pass,
    restore_after_regular_pass,
    inject_for_pml_pass,
    restore_after_pml_pass,
    _sub_pml_sides,
    _op_left_pml, _op_right_pml, _op_bottom_pml,
    _op_left_reg, _op_right_reg, _op_bottom_reg,
)

@dataclass
class HybridResult:
    """Result of a hybrid FD-FK simulation.

    Attributes
    ----------
    t : ndarray, shape (nt,)
        Time axis (s).
    ux, uz : ndarray, shape (nrcv, nt)
        Horizontal and vertical displacement seismograms.
    receivers_x, receivers_z : ndarray
        Receiver coordinates (m).
    config : HybridConfig
        Simulation parameters used.
    """

    t: np.ndarray
    ux: np.ndarray
    uz: np.ndarray
    receivers_x: np.ndarray
    receivers_z: np.ndarray
    config: HybridConfig


def _fd_update_regular(solver: FDDisplacementSolver, config: HybridConfig):
    """FD update for regular-domain nodes only."""
    fd_update_regular(solver)


def _fd_update_pml(solver: FDDisplacementSolver, config: HybridConfig):
    """FD update for PML-domain nodes only (left, right, bottom)."""
    fd_update_pml(solver)


def _apply_pml_only(solver: FDDisplacementSolver):
    """Apply PML damping to ux_new/uz_new (Sochacki or C-PML)."""
    if solver.use_cpml:
        _apply_cpml_damping(
            solver.ux_new, solver.uz_new,
            solver.ux_cur, solver.ux_old,
            solver.uz_cur, solver.uz_old,
            solver.dt, solver.dx_deff, solver.dz_deff,
            solver.dx_a, solver.dz_a, solver.dx_b, solver.dz_b,
            solver.psi_ux, solver.psi_uz,
            solver.nz_tot, solver.nx_tot, solver.npml, solver.nz, solver.nx,
        )
    else:
        _apply_pml_damping(
            solver.ux_new, solver.uz_new,
            solver.ux_cur, solver.ux_old,
            solver.uz_cur, solver.uz_old,
            solver.dt, solver.dx_damp, solver.dz_damp,
            solver.nz_tot, solver.nx_tot, solver.npml, solver.norder,
        )


def _apply_free_surface_only(solver: FDDisplacementSolver):
    """Apply free-surface BC to ux_new/uz_new."""
    _apply_free_surface(
        solver.ux_new, solver.uz_new,
        solver.lam, solver.mu, solver.norder, solver.coef,
        solver.dz, solver.iz_surf, solver.nx_tot, solver.dx,
    )


def _restore_pml_cur_old(solver, bwf, config, it):
    """Subtract u0 from PML u_cur and u_old (restore to scattered)."""
    nbd = config.nbd
    npml = config.npml
    nx, nz = config.nx, config.nz
    nt_bwf = bwf.ux_left.shape[-1]

    if 0 <= it < nt_bwf:
        _sub_pml_sides(solver.ux_cur, solver.uz_cur, bwf, nbd, npml, nx, nz, it)
    it_old = max(it - 1, 0)
    if it_old < nt_bwf:
        _sub_pml_sides(solver.ux_old, solver.uz_old, bwf, nbd, npml, nx, nz, it_old)


# ===================================================================
# Fused Numba time-stepping kernel
# ===================================================================

@njit(cache=True)
def _inject_add_pml(ux, uz, bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                    bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it):
    """Add u0 to PML coupling-zone nodes (all three boundaries)."""
    _op_left_pml(ux, uz, bwf_ux_l[:, :nbd, it], bwf_uz_l[:, :nbd, it],
                 nbd, npml, nz, True)
    _op_right_pml(ux, uz, bwf_ux_r[:, nbd:, it], bwf_uz_r[:, nbd:, it],
                  nbd, npml, nx, nz, True)
    _op_bottom_pml(ux, uz, bwf_ux_b[:, nbd:, it], bwf_uz_b[:, nbd:, it],
                   nbd, npml, nx, nz, True)


@njit(cache=True)
def _inject_sub_pml(ux, uz, bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                    bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it):
    """Subtract u0 from PML coupling-zone nodes."""
    _op_left_pml(ux, uz, bwf_ux_l[:, :nbd, it], bwf_uz_l[:, :nbd, it],
                 nbd, npml, nz, False)
    _op_right_pml(ux, uz, bwf_ux_r[:, nbd:, it], bwf_uz_r[:, nbd:, it],
                  nbd, npml, nx, nz, False)
    _op_bottom_pml(ux, uz, bwf_ux_b[:, nbd:, it], bwf_uz_b[:, nbd:, it],
                   nbd, npml, nx, nz, False)


@njit(cache=True)
def _inject_sub_reg(ux, uz, bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                    bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it):
    """Subtract u0 from regular coupling-zone nodes.

    Left/right cover only the top nz-nbd rows; the bottom nbd rows are
    handled exclusively by bottom injection to avoid corner overlap.
    Bottom covers only regular-domain x-columns (matching Fortran).
    """
    nz_top = nz - nbd
    _op_left_reg(ux, uz, bwf_ux_l[:nz_top, nbd:nbd + norder, it],
                 bwf_uz_l[:nz_top, nbd:nbd + norder, it],
                 nbd, norder, npml, nz, False)
    _op_right_reg(ux, uz, bwf_ux_r[:nz_top, nbd - norder:nbd, it],
                  bwf_uz_r[:nz_top, nbd - norder:nbd, it],
                  nbd, norder, npml, nx, nz, False)
    _op_bottom_reg(ux, uz, bwf_ux_b[nbd:nbd + nx, :nbd, it],
                   bwf_uz_b[nbd:nbd + nx, :nbd, it],
                   nbd, norder, npml, nx, nz, False)


@njit(cache=True)
def _inject_add_reg(ux, uz, bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                    bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it):
    """Add u0 back to regular coupling-zone nodes.

    Left/right cover only the top nz-nbd rows; the bottom nbd rows are
    handled exclusively by bottom injection to avoid corner overlap.
    Bottom covers only regular-domain x-columns (matching Fortran).
    """
    nz_top = nz - nbd
    _op_left_reg(ux, uz, bwf_ux_l[:nz_top, nbd:nbd + norder, it],
                 bwf_uz_l[:nz_top, nbd:nbd + norder, it],
                 nbd, norder, npml, nz, True)
    _op_right_reg(ux, uz, bwf_ux_r[:nz_top, nbd - norder:nbd, it],
                  bwf_uz_r[:nz_top, nbd - norder:nbd, it],
                  nbd, norder, npml, nx, nz, True)
    _op_bottom_reg(ux, uz, bwf_ux_b[nbd:nbd + nx, :nbd, it],
                   bwf_uz_b[nbd:nbd + nx, :nbd, it],
                   nbd, norder, npml, nx, nz, True)


@njit(cache=True)
def _store_wavefield_snapshot(ux, uz, snap_ux, snap_uz, snap_idx, nz, nx, npml):
    """Copy regular-domain displacement into preallocated snapshot buffers."""
    for iz in range(nz):
        ix_off = npml
        for ix in range(nx):
            snap_ux[snap_idx, iz, ix] = ux[iz, ix + ix_off]
            snap_uz[snap_idx, iz, ix] = uz[iz, ix + ix_off]


@njit(cache=True)
def _run_timeloop_numba(
    ux_buf0, uz_buf0, ux_buf1, uz_buf1, ux_buf2, uz_buf2,
    lam2mu, mu_raw, lam_raw, rho,
    coef0, coef1, norder,
    lam, mu,
    dt, dx, dz, nx, nz, npml,
    nz_tot, nx_tot, iz_start,
    dx_damp, dz_damp,
    bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r, bwf_ux_b, bwf_uz_b,
    nbd,
    rcv_ix, rcv_iz,
    seis_ux, seis_uz,
    nt,
    snapshot_interval,
    snap_ux, snap_uz,
):
    """Fused conservative-form time loop (Fortran-matching coef0/coef1).

    Eliminates all Python-level overhead between time steps by running the
    entire loop inside a single Numba-compiled function.

    The three buffer pairs (buf0, buf1, buf2) are cycled via modular index
    to implement the old/cur/new time-level rotation without array copies.

    When ``snapshot_interval > 0``, wavefield snapshots are written to
    ``snap_ux`` / ``snap_uz`` (shape ``(nsnap, nz, nx)``) at the same times
    as the legacy Python step loop.
    """
    nt_bwf = bwf_ux_l.shape[2]
    nrcv = rcv_ix.shape[0]

    for it in range(nt):
        # --- Select time-level buffers via cyclic rotation ---
        r = it % 3
        if r == 0:
            ux_old = ux_buf0; uz_old = uz_buf0
            ux_cur = ux_buf1; uz_cur = uz_buf1
            ux_new = ux_buf2; uz_new = uz_buf2
        elif r == 1:
            ux_old = ux_buf1; uz_old = uz_buf1
            ux_cur = ux_buf2; uz_cur = uz_buf2
            ux_new = ux_buf0; uz_new = uz_buf0
        else:
            ux_old = ux_buf2; uz_old = uz_buf2
            ux_cur = ux_buf0; uz_cur = uz_buf0
            ux_new = ux_buf1; uz_new = uz_buf1

        it_old = it - 1
        if it_old < 0:
            it_old = 0

        # === Pass 1: Regular-domain FD update ===
        if it < nt_bwf:
            _inject_add_pml(ux_cur, uz_cur,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it)
        if it_old < nt_bwf:
            _inject_add_pml(ux_old, uz_old,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it_old)

        # Boundary rows near free surface (variable stencil order)
        _update_displacement_conservative(
            ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
            lam2mu, mu_raw, lam_raw, rho,
            coef0, coef1, norder, dt, dx, dz,
            iz_start, nz, npml, npml + nx,
        )

        # Restore PML to scattered field
        if it < nt_bwf:
            _inject_sub_pml(ux_cur, uz_cur,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it)
        if it_old < nt_bwf:
            _inject_sub_pml(ux_old, uz_old,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it_old)

        # Convert u_new in PML to scattered (subtract u0^{n+1}); matches
        # restore_after_regular_pass() in the Python step loop.
        it_new = it + 1
        if 0 <= it_new < nt_bwf:
            _inject_sub_pml(ux_new, uz_new,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it_new)

        # === Pass 2: PML-domain FD update ===
        if it < nt_bwf:
            _inject_sub_reg(ux_cur, uz_cur,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it)
        if it_old < nt_bwf:
            _inject_sub_reg(ux_old, uz_old,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it_old)

        if npml > 1:
            _update_displacement_conservative(
                ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                lam2mu, mu_raw, lam_raw, rho,
                coef0, coef1, norder, dt, dx, dz,
                1, nz_tot, 1, npml,
            )
        if npml + nx < nx_tot:
            _update_displacement_conservative(
                ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                lam2mu, mu_raw, lam_raw, rho,
                coef0, coef1, norder, dt, dx, dz,
                1, nz_tot, npml + nx, nx_tot - 1,
            )
        if nz < nz_tot:
            _update_displacement_conservative(
                ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                lam2mu, mu_raw, lam_raw, rho,
                coef0, coef1, norder, dt, dx, dz,
                nz, nz_tot, npml, npml + nx,
            )

        # Restore regular coupling nodes to total field
        if it < nt_bwf:
            _inject_add_reg(ux_cur, uz_cur,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it)
        if it_old < nt_bwf:
            _inject_add_reg(ux_old, uz_old,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it_old)

        # === PML damping ===
        _apply_pml_damping(
            ux_new, uz_new, ux_cur, ux_old, uz_cur, uz_old,
            dt, dx_damp, dz_damp, nz_tot, nx_tot, npml, norder,
        )

        # === Free-surface BC ===
        _apply_free_surface(ux_new, uz_new, lam, mu, norder, coef1,
                            dz, 0, nx_tot, dx)

        if snapshot_interval > 0:
            if it % snapshot_interval == 0:
                snap_idx = it // snapshot_interval
                _store_wavefield_snapshot(
                    ux_new, uz_new, snap_ux, snap_uz, snap_idx, nz, nx, npml,
                )

        # === Record seismograms from ux_new (becomes ux_cur after rotation) ===
        for j in range(nrcv):
            seis_ux[j, it] = ux_new[rcv_iz[j], rcv_ix[j]]
            seis_uz[j, it] = uz_new[rcv_iz[j], rcv_ix[j]]


@njit(cache=True)
def _run_timeloop_numba_legacy(
    ux_buf0, uz_buf0, ux_buf1, uz_buf1, ux_buf2, uz_buf2,
    lam2mu_over_rho, mu_over_rho, lam_mu_over_rho,
    d2_coef_x, d2_coef_z,
    lam, mu, coef, norder,
    dt, dx, dz, nx, nz, npml,
    nz_tot, nx_tot, iz_start,
    dx_damp, dz_damp,
    bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r, bwf_ux_b, bwf_uz_b,
    nbd,
    rcv_ix, rcv_iz,
    seis_ux, seis_uz,
    nt,
):
    """Legacy non-conservative fused loop (``conservative=False`` only)."""
    nt_bwf = bwf_ux_l.shape[2]
    nrcv = rcv_ix.shape[0]

    for it in range(nt):
        r = it % 3
        if r == 0:
            ux_old = ux_buf0; uz_old = uz_buf0
            ux_cur = ux_buf1; uz_cur = uz_buf1
            ux_new = ux_buf2; uz_new = uz_buf2
        elif r == 1:
            ux_old = ux_buf1; uz_old = uz_buf1
            ux_cur = ux_buf2; uz_cur = uz_buf2
            ux_new = ux_buf0; uz_new = uz_buf0
        else:
            ux_old = ux_buf2; uz_old = uz_buf2
            ux_cur = ux_buf0; uz_cur = uz_buf0
            ux_new = ux_buf1; uz_new = uz_buf1

        it_old = it - 1
        if it_old < 0:
            it_old = 0

        if it < nt_bwf:
            _inject_add_pml(ux_cur, uz_cur,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it)
        if it_old < nt_bwf:
            _inject_add_pml(ux_old, uz_old,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it_old)

        iz_bnd = min(norder, nz)
        if iz_start < iz_bnd:
            _update_displacement(
                ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                lam2mu_over_rho, mu_over_rho, lam_mu_over_rho,
                d2_coef_x, d2_coef_z,
                coef, norder, dt, dx, dz,
                iz_start, iz_bnd, npml, npml + nx,
            )
        iz_int = max(iz_start, norder)
        if iz_int < nz:
            _update_displacement_full(
                ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                lam2mu_over_rho, mu_over_rho, lam_mu_over_rho,
                d2_coef_x, d2_coef_z,
                coef, norder, dt, dx, dz,
                iz_int, nz, npml, npml + nx,
            )

        if it < nt_bwf:
            _inject_sub_pml(ux_cur, uz_cur,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it)
        if it_old < nt_bwf:
            _inject_sub_pml(ux_old, uz_old,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it_old)

        it_new = it + 1
        if 0 <= it_new < nt_bwf:
            _inject_sub_pml(ux_new, uz_new,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it_new)

        if it < nt_bwf:
            _inject_sub_reg(ux_cur, uz_cur,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it)
        if it_old < nt_bwf:
            _inject_sub_reg(ux_old, uz_old,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it_old)

        if npml > 1:
            _update_displacement(
                ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                lam2mu_over_rho, mu_over_rho, lam_mu_over_rho,
                d2_coef_x, d2_coef_z,
                coef, norder, dt, dx, dz,
                1, nz_tot, 1, npml,
            )
        if npml + nx < nx_tot:
            _update_displacement(
                ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                lam2mu_over_rho, mu_over_rho, lam_mu_over_rho,
                d2_coef_x, d2_coef_z,
                coef, norder, dt, dx, dz,
                1, nz_tot, npml + nx, nx_tot - 1,
            )
        if nz < nz_tot:
            _update_displacement(
                ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                lam2mu_over_rho, mu_over_rho, lam_mu_over_rho,
                d2_coef_x, d2_coef_z,
                coef, norder, dt, dx, dz,
                nz, nz_tot, npml, npml + nx,
            )

        if it < nt_bwf:
            _inject_add_reg(ux_cur, uz_cur,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it)
        if it_old < nt_bwf:
            _inject_add_reg(ux_old, uz_old,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it_old)

        _apply_pml_damping(
            ux_new, uz_new, ux_cur, ux_old, uz_cur, uz_old,
            dt, dx_damp, dz_damp, nz_tot, nx_tot, npml, norder,
        )

        _apply_free_surface(ux_new, uz_new, lam, mu, norder, coef,
                            dz, 0, nx_tot, dx)

        for j in range(nrcv):
            seis_ux[j, it] = ux_new[rcv_iz[j], rcv_ix[j]]
            seis_uz[j, it] = uz_new[rcv_iz[j], rcv_ix[j]]


@njit(cache=True)
def _run_timeloop_numba_cpml(
    ux_buf0, uz_buf0, ux_buf1, uz_buf1, ux_buf2, uz_buf2,
    lam2mu, mu_raw, lam_raw, rho,
    coef0, coef1, norder,
    lam, mu,
    dt, dx, dz, nx, nz, npml,
    nz_tot, nx_tot, iz_start,
    dx_deff, dz_deff, dx_a, dz_a, dx_b, dz_b,
    psi_ux, psi_uz,
    bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r, bwf_ux_b, bwf_uz_b,
    nbd,
    rcv_ix, rcv_iz,
    seis_ux, seis_uz,
    nt,
    snapshot_interval,
    snap_ux, snap_uz,
):
    """Fused conservative-form time loop with C-PML damping."""
    nt_bwf = bwf_ux_l.shape[2]
    nrcv = rcv_ix.shape[0]

    for it in range(nt):
        r = it % 3
        if r == 0:
            ux_old = ux_buf0; uz_old = uz_buf0
            ux_cur = ux_buf1; uz_cur = uz_buf1
            ux_new = ux_buf2; uz_new = uz_buf2
        elif r == 1:
            ux_old = ux_buf1; uz_old = uz_buf1
            ux_cur = ux_buf2; uz_cur = uz_buf2
            ux_new = ux_buf0; uz_new = uz_buf0
        else:
            ux_old = ux_buf2; uz_old = uz_buf2
            ux_cur = ux_buf0; uz_cur = uz_buf0
            ux_new = ux_buf1; uz_new = uz_buf1

        it_old = it - 1
        if it_old < 0:
            it_old = 0

        # === Pass 1: Regular-domain FD update ===
        if it < nt_bwf:
            _inject_add_pml(ux_cur, uz_cur,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it)
        if it_old < nt_bwf:
            _inject_add_pml(ux_old, uz_old,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it_old)

        _update_displacement_conservative(
            ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
            lam2mu, mu_raw, lam_raw, rho,
            coef0, coef1, norder, dt, dx, dz,
            iz_start, nz, npml, npml + nx,
        )

        # Restore PML to scattered field
        if it < nt_bwf:
            _inject_sub_pml(ux_cur, uz_cur,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it)
        if it_old < nt_bwf:
            _inject_sub_pml(ux_old, uz_old,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it_old)

        it_new = it + 1
        if 0 <= it_new < nt_bwf:
            _inject_sub_pml(ux_new, uz_new,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, npml, nx, nz, it_new)

        # === Pass 2: PML-domain FD update ===
        if it < nt_bwf:
            _inject_sub_reg(ux_cur, uz_cur,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it)
        if it_old < nt_bwf:
            _inject_sub_reg(ux_old, uz_old,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it_old)

        if npml > 1:
            _update_displacement_conservative(
                ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                lam2mu, mu_raw, lam_raw, rho,
                coef0, coef1, norder, dt, dx, dz,
                1, nz_tot, 1, npml,
            )
        if npml + nx < nx_tot:
            _update_displacement_conservative(
                ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                lam2mu, mu_raw, lam_raw, rho,
                coef0, coef1, norder, dt, dx, dz,
                1, nz_tot, npml + nx, nx_tot - 1,
            )
        if nz < nz_tot:
            _update_displacement_conservative(
                ux_new, uz_new, ux_cur, uz_cur, ux_old, uz_old,
                lam2mu, mu_raw, lam_raw, rho,
                coef0, coef1, norder, dt, dx, dz,
                nz, nz_tot, npml, npml + nx,
            )

        # Restore regular coupling nodes to total field
        if it < nt_bwf:
            _inject_add_reg(ux_cur, uz_cur,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it)
        if it_old < nt_bwf:
            _inject_add_reg(ux_old, uz_old,
                            bwf_ux_l, bwf_uz_l, bwf_ux_r, bwf_uz_r,
                            bwf_ux_b, bwf_uz_b, nbd, norder, npml, nx, nz, it_old)

        # === C-PML damping ===
        _apply_cpml_damping(
            ux_new, uz_new, ux_cur, ux_old, uz_cur, uz_old,
            dt, dx_deff, dz_deff, dx_a, dz_a, dx_b, dz_b,
            psi_ux, psi_uz,
            nz_tot, nx_tot, npml, nz, nx,
        )

        # === Free-surface BC ===
        _apply_free_surface(ux_new, uz_new, lam, mu, norder, coef1,
                            dz, 0, nx_tot, dx)

        if snapshot_interval > 0:
            if it % snapshot_interval == 0:
                snap_idx = it // snapshot_interval
                _store_wavefield_snapshot(
                    ux_new, uz_new, snap_ux, snap_uz, snap_idx, nz, nx, npml,
                )

        # === Record seismograms ===
        for j in range(nrcv):
            seis_ux[j, it] = ux_new[rcv_iz[j], rcv_ix[j]]
            seis_uz[j, it] = uz_new[rcv_iz[j], rcv_ix[j]]


def _init_coupling_nodes(solver, bwf, config):
    """Pre-populate regular coupling nodes with bwf(t=0).

    At the start of the simulation, u_cur and u_old are zero everywhere.
    The TF/SF injection subtracts bwf from regular coupling nodes to get
    the scattered field, but scattered = 0 - bwf = -bwf, creating a
    spurious wave. By pre-setting coupling nodes to bwf(0), the
    subtraction yields scattered = bwf(0) - bwf(0) = 0.

    We also pre-populate u_old with bwf(0) as an approximation for
    bwf(t=-dt), since the bwf array does not cover negative times.
    """
    from .injection import _add_reg_sides
    nbd = config.nbd
    norder = config.norder
    npml = config.npml
    nx, nz = config.nx, config.nz

    _add_reg_sides(solver.ux_cur, solver.uz_cur, bwf,
                   nbd, norder, npml, nx, nz, 0)
    _add_reg_sides(solver.ux_old, solver.uz_old, bwf,
                   nbd, norder, npml, nx, nz, 0)


def _dispatch_timeloop(solver, config, bwf, rcv_ix, rcv_iz,
                       seis_ux, seis_uz, nt, *,
                       snapshot_interval: int = 0,
                       snap_ux=None, snap_uz=None):
    """Choose fused Numba time loop (conservative or legacy)."""
    snap_iv = snapshot_interval
    if snap_ux is None:
        snap_ux = np.empty((1, 1, 1), dtype=np.float64)
    if snap_uz is None:
        snap_uz = np.empty((1, 1, 1), dtype=np.float64)

    if not solver.conservative:
        if solver.use_cpml:
            raise NotImplementedError(
                "conservative=False with C-PML is not supported in the fused loop"
            )
        _run_timeloop_numba_legacy(
            solver.ux_old, solver.uz_old,
            solver.ux_cur, solver.uz_cur,
            solver.ux_new, solver.uz_new,
            solver.lam2mu_over_rho, solver.mu_over_rho, solver.lam_mu_over_rho,
            solver.d2_coef_x, solver.d2_coef_z,
            solver.lam, solver.mu,
            solver.coef, solver.norder,
            solver.dt, solver.dx, solver.dz,
            config.nx, config.nz, config.npml,
            solver.nz_tot, solver.nx_tot, solver.iz_start,
            solver.dx_damp, solver.dz_damp,
            bwf.ux_left, bwf.uz_left,
            bwf.ux_right, bwf.uz_right,
            bwf.ux_bottom, bwf.uz_bottom,
            config.nbd,
            rcv_ix, rcv_iz,
            seis_ux, seis_uz,
            nt,
        )
        return

    if solver.use_cpml:
        _run_timeloop_numba_cpml(
            solver.ux_old, solver.uz_old,
            solver.ux_cur, solver.uz_cur,
            solver.ux_new, solver.uz_new,
            solver.lam2mu, solver.mu_raw, solver.lam_raw, solver.rho,
            solver.coef0, solver.coef,
            solver.norder,
            solver.lam, solver.mu,
            solver.dt, solver.dx, solver.dz,
            config.nx, config.nz, config.npml,
            solver.nz_tot, solver.nx_tot, solver.iz_start,
            solver.dx_deff, solver.dz_deff,
            solver.dx_a, solver.dz_a, solver.dx_b, solver.dz_b,
            solver.psi_ux, solver.psi_uz,
            bwf.ux_left, bwf.uz_left,
            bwf.ux_right, bwf.uz_right,
            bwf.ux_bottom, bwf.uz_bottom,
            config.nbd,
            rcv_ix, rcv_iz,
            seis_ux, seis_uz,
            nt,
            snap_iv, snap_ux, snap_uz,
        )
    else:
        _run_timeloop_numba(
            solver.ux_old, solver.uz_old,
            solver.ux_cur, solver.uz_cur,
            solver.ux_new, solver.uz_new,
            solver.lam2mu, solver.mu_raw, solver.lam_raw, solver.rho,
            solver.coef0, solver.coef,
            solver.norder,
            solver.lam, solver.mu,
            solver.dt, solver.dx, solver.dz,
            config.nx, config.nz, config.npml,
            solver.nz_tot, solver.nx_tot, solver.iz_start,
            solver.dx_damp, solver.dz_damp,
            bwf.ux_left, bwf.uz_left,
            bwf.ux_right, bwf.uz_right,
            bwf.ux_bottom, bwf.uz_bottom,
            config.nbd,
            rcv_ix, rcv_iz,
            seis_ux, seis_uz,
            nt,
            snap_iv, snap_ux, snap_uz,
        )


def compute_hybrid_seismograms(
    config: HybridConfig,
    model_fk_left: FKLayerModel,
    model_fk_right: FKLayerModel,
    source: RickerSource,
    p_si: float,
    receivers_x: np.ndarray | Sequence[float],
    receivers_z: np.ndarray | Sequence[float] | None = None,
    *,
    vp_2d: np.ndarray | None = None,
    vs_2d: np.ndarray | None = None,
    rho_2d: np.ndarray | None = None,
    phi: float = 0.0,
    z_init: float | None = None,
    freq_damping: float = 0.0,
    bwf_pad_factor: float = 1.0,
    fortran_seisx_convention: bool = True,
    verbose: bool = True,
    snapshot_interval: int = 0,
) -> HybridResult:
    """Run a full hybrid FD-FK simulation and extract seismograms.

    Parameters
    ----------
    config : HybridConfig
        Grid, time-stepping, and PML parameters.
    model_fk_left, model_fk_right : FKLayerModel
        1D velocity models for the left and right boundaries.
        For a laterally homogeneous model, pass the same object for both.
    source : RickerSource
        Source wavelet and incidence type (P or S).
    p_si : float
        Horizontal slowness (s/m).
    receivers_x : array_like
        Receiver x-coordinates (m) within the regular domain.
    receivers_z : array_like, optional
        Receiver z-coordinates (m). Default: all at the surface (z=0).
    vp_2d, vs_2d, rho_2d : ndarray, shape (nz, nx), optional
        2D velocity and density model for the FD domain. If None,
        the model is built from model_fk_left (laterally homogeneous).
    phi : float
        Back-azimuth angle (rad) for the plane-wave moveout.
    z_init : float, optional
        Plane-wave initialization depth (m) in the half-space.
    freq_damping : float
        FK frequency damping (rad/s). 0 disables.
    bwf_pad_factor : float
        Anti-wraparound zero-padding factor for the FK boundary-wavefield
        synthesis.  ``1.0`` (default) = OFF; ``> 1.0`` (e.g. ``2.0``) = ON.
    fortran_seisx_convention : bool
        Apply an extra ``cos(phi)`` post-rotation to the recorded ``ux``
        seismograms so that ``result.ux`` matches the Fortran ``seisx.su``
        output convention.
    verbose : bool
        Print progress messages.
    snapshot_interval : int
        If > 0, store full wavefield snapshots every N steps. Uses the same
        fused Numba time loop as seismogram-only runs (no slow Python path).

    Returns
    -------
    HybridResult
    """
    receivers_x = np.atleast_1d(np.asarray(receivers_x, dtype=float))
    nrcv = receivers_x.size
    if receivers_z is None:
        receivers_z = np.zeros(nrcv)
    receivers_z = np.atleast_1d(np.asarray(receivers_z, dtype=float))

    nx, nz = config.nx, config.nz
    nt = config.nt

    # --- Build 2D material model ---
    if vp_2d is None:
        vp_2d, vs_2d, rho_2d = _model_from_fk(model_fk_left, config)

    # Anti-wraparound synthesis length
    nt_syn = synthesis_length(nt, bwf_pad_factor)

    if verbose:
        print(f"[Hybrid] Computing FK background wavefield "
              f"(nz={nz}, nx={nx}, nt={nt})...")

    bwf = compute_boundary_wavefield(
        config, model_fk_left, model_fk_right, source, p_si,
        phi=phi, z_init=z_init, freq_damping=freq_damping, nt_syn=nt_syn,
    )

    if verbose:
        mem_mb = (bwf.ux_left.nbytes + bwf.uz_left.nbytes +
                  bwf.ux_right.nbytes + bwf.uz_right.nbytes +
                  bwf.ux_bottom.nbytes + bwf.uz_bottom.nbytes) / 1e6
        print(f"[Hybrid] Background wavefield memory: {mem_mb:.1f} MB")

    # --- Set up FD solver ---
    if verbose:
        print("[Hybrid] Initializing FD solver...")

    rho_ext, lam_ext, mu_ext = build_material_arrays(config, vp_2d, vs_2d, rho_2d)
    solver = FDDisplacementSolver(config, rho_ext, lam_ext, mu_ext)

    # --- Time-stepping ---
    if verbose:
        print(f"[Hybrid] Time-stepping: {nt} steps, dt={config.dt:.5f}s...")

    seis_ux = np.zeros((nrcv, nt), dtype=np.float32)
    seis_uz = np.zeros((nrcv, nt), dtype=np.float32)

    rcv_ix_ext, rcv_iz_ext = _receiver_indices(
        receivers_x, receivers_z, config
    )

    _init_coupling_nodes(solver, bwf, config)

    snapshots_ux = []
    snapshots_uz = []

    nsnap = 0
    snap_ux = snap_uz = None
    if snapshot_interval > 0:
        nsnap = (nt - 1) // snapshot_interval + 1
        snap_ux = np.zeros((nsnap, nz, nx), dtype=np.float32)
        snap_uz = np.zeros((nsnap, nz, nx), dtype=np.float32)
        if verbose:
            mem_mb = 2 * snap_ux.nbytes / 1e6
            print(
                f"[Hybrid] Snapshot storage: {nsnap} frames, "
                f"{mem_mb:.1f} MB (Numba fused loop)"
            )

    if not solver.conservative and snapshot_interval > 0:
        # Legacy non-conservative form: step-by-step Python loop with snapshots.
        for it in range(nt):
            inject_for_regular_pass(solver, bwf, config, it)
            _fd_update_regular(solver, config)
            restore_after_regular_pass(solver, bwf, config, it)
            inject_for_pml_pass(solver, bwf, config, it)
            _fd_update_pml(solver, config)
            restore_after_pml_pass(solver, bwf, config, it)
            _apply_pml_only(solver)
            _apply_free_surface_only(solver)
            solver.ux_old, solver.ux_cur, solver.ux_new = (
                solver.ux_cur, solver.ux_new, solver.ux_old)
            solver.uz_old, solver.uz_cur, solver.uz_new = (
                solver.uz_cur, solver.uz_new, solver.uz_old)
            for j in range(nrcv):
                seis_ux[j, it] = solver.ux_cur[rcv_iz_ext[j], rcv_ix_ext[j]]
                seis_uz[j, it] = solver.uz_cur[rcv_iz_ext[j], rcv_ix_ext[j]]
            if it % snapshot_interval == 0:
                ux_snap, uz_snap = solver.get_displacement()
                snapshots_ux.append(ux_snap.copy())
                snapshots_uz.append(uz_snap.copy())
    else:
        _dispatch_timeloop(
            solver, config, bwf,
            rcv_ix_ext, rcv_iz_ext,
            seis_ux, seis_uz, nt,
            snapshot_interval=snapshot_interval,
            snap_ux=snap_ux,
            snap_uz=snap_uz,
        )

    if fortran_seisx_convention and phi != 0.0:
        seis_ux *= np.float32(np.cos(phi))

    if verbose:
        print("[Hybrid] Simulation complete.")

    t = np.arange(nt) * config.dt

    result = HybridResult(
        t=t,
        ux=seis_ux,
        uz=seis_uz,
        receivers_x=receivers_x,
        receivers_z=receivers_z,
        config=config,
    )

    if snapshot_interval > 0:
        if snapshots_ux:
            result.snapshots_ux = snapshots_ux
            result.snapshots_uz = snapshots_uz
        else:
            result.snapshots_ux = snap_ux
            result.snapshots_uz = snap_uz

    return result


def _model_from_fk(model: FKLayerModel, config: HybridConfig):
    """Build 2D arrays from a 1D FK model (laterally homogeneous).

    Returns vp_2d, vs_2d, rho_2d of shape (nz, nx).
    """
    nz, nx = config.nz, config.nx
    z_coords = config.z_coords

    vp_2d = np.zeros((nz, nx), dtype=np.float64)
    vs_2d = np.zeros((nz, nx), dtype=np.float64)
    rho_2d = np.zeros((nz, nx), dtype=np.float64)

    for iz, z in enumerate(z_coords):
        # Find which layer this depth belongs to
        layer_idx = np.searchsorted(model.z_top, z, side="right") - 1
        layer_idx = max(0, min(layer_idx, model.halfspace_index))
        vp_2d[iz, :] = model.vp[layer_idx]
        vs_2d[iz, :] = model.vs[layer_idx]
        rho_2d[iz, :] = model.rho[layer_idx]

    return vp_2d, vs_2d, rho_2d


def _receiver_indices(receivers_x, receivers_z, config: HybridConfig):
    """Convert receiver coordinates to extended-grid indices.

    Returns ix_ext, iz_ext arrays (nearest grid point).
    Surface receivers (iz=0) are placed at iz=1 because the FD update
    starts at iz=1 and iz=0 is set by free-surface extrapolation.
    """
    npml = config.npml
    nrcv = receivers_x.size

    ix_ext = np.zeros(nrcv, dtype=int)
    iz_ext = np.zeros(nrcv, dtype=int)

    for j in range(nrcv):
        ix_reg = int(round((receivers_x[j] - config.x0) / config.dx))
        iz_reg = int(round((receivers_z[j] - config.z0) / config.dz))
        ix_reg = max(0, min(ix_reg, config.nx - 1))
        iz_reg = max(0, min(iz_reg, config.nz - 1))
        # Ensure receivers are at least at iz=1 (iz=0 is free-surface ghost)
        iz_reg = max(iz_reg, 1)
        ix_ext[j] = ix_reg + npml
        iz_ext[j] = iz_reg

    return ix_ext, iz_ext


# ===================================================================
# Batch-friendly engine (reuses FK / material between runs)
# ===================================================================

class HybridEngine:
    """Reusable hybrid FD-FK engine for batch forward modeling.

    Separates one-time setup (FK computation, Numba JIT compilation) from
    per-run execution.  For N=1000 forward runs with fixed ray parameter
    and 1D boundary models, the FK propagator matrices are computed once
    and the boundary wavefield is built cheaply for each source wavelet.

    Typical usage::

        engine = HybridEngine(config, model_fk, model_fk, source, p_si)
        for model_2d in models:
            result = engine.run(receivers_x, vp_2d=model_2d.vp, ...)

    Parameters
    ----------
    config : HybridConfig
    model_fk_left, model_fk_right : FKLayerModel
    source : RickerSource
    p_si : float
    phi, z_init, freq_damping : float
        Forwarded to FK computation.
    bwf_pad_factor : float
        Anti-wraparound zero-pad factor for the FK boundary-wavefield
        synthesis, applied once when the Green's-function cache is built.

        * ``1.0`` (default) = OFF: synthesise at exactly ``config.nt`` (the
          original behaviour, no extra cost).
        * ``> 1.0`` (e.g. ``2.0``) = ON: synthesise on ~``pad_factor``x the
          time axis (rounded to an FFT-friendly length by
          :func:`hybrid.background.synthesis_length`) and truncate back to
          ``nt``.

        Problem solved: removes spurious early-time energy caused by IFFT
        time-wraparound when the latest boundary arrival exceeds the record
        length ``nt * dt``. Cost: increases only the one-time FK synthesis
        (≈ ``nt_syn / nt``) and transient memory; the per-run FD time-stepping
        cost is unchanged. See :meth:`__init__` / ``compute_hybrid_seismograms``
        for the full description and the honest no-op note for test1-P.
    fortran_seisx_convention : bool
        If ``True`` (default), multiply recorded ``ux`` seismograms by an
        extra ``cos(phi)`` at output to match the Fortran ``seisx.su``
        double-cos convention. See :func:`compute_hybrid_seismograms` for
        the full explanation. Only affects ux; uz is unchanged.
    """

    def __init__(
        self,
        config: HybridConfig,
        model_fk_left: FKLayerModel,
        model_fk_right: FKLayerModel,
        source: RickerSource,
        p_si: float,
        *,
        phi: float = 0.0,
        z_init: float | None = None,
        freq_damping: float = 0.0,
        bwf_pad_factor: float = 1.0,
        fortran_seisx_convention: bool = True,
        verbose: bool = True,
    ):
        self.config = config
        self.source = source
        self.p_si = p_si
        self.phi = phi
        self.freq_damping = freq_damping
        self.fortran_seisx_convention = fortran_seisx_convention

        # Anti-wraparound synthesis length for the boundary wavefield
        nt_syn = synthesis_length(config.nt, bwf_pad_factor)

        # Pre-compute FK Green's functions (source-independent)
        if verbose:
            print("[HybridEngine] Computing FK Green's functions...")
        self._greens = compute_fk_greens(
            config, model_fk_left, model_fk_right, p_si,
            source.incidence, phi=phi, z_init=z_init, nt_syn=nt_syn,
        )

        # Build boundary wavefield for the initial source
        if verbose:
            print("[HybridEngine] Building boundary wavefield...")
        self._bwf = boundary_wavefield_from_greens(
            config, self._greens, source, freq_damping=freq_damping,
        )

        # Default 2D material from left FK model
        self._vp_2d, self._vs_2d, self._rho_2d = _model_from_fk(
            model_fk_left, config
        )

        # Pre-build solver with default material (reused across runs)
        rho_ext, lam_ext, mu_ext = build_material_arrays(
            config, self._vp_2d, self._vs_2d, self._rho_2d
        )
        self._solver = FDDisplacementSolver(config, rho_ext, lam_ext, mu_ext)

        self._compiled = False
        if verbose:
            print("[HybridEngine] Ready.")

    @property
    def greens_cache(self) -> FKGreensCache:
        """The cached source-independent FK Green's functions."""
        return self._greens

    def update_source(self, source: RickerSource) -> None:
        """Rebuild boundary wavefield for a new source wavelet.

        Only performs the cheap spectrum multiplication + IFFT; the
        expensive FK propagator-matrix computation is skipped.
        """
        self.source = source
        self._bwf = boundary_wavefield_from_greens(
            self.config, self._greens, source,
            freq_damping=self.freq_damping,
        )

    def run(
        self,
        receivers_x: np.ndarray | Sequence[float],
        receivers_z: np.ndarray | Sequence[float] | None = None,
        *,
        vp_2d: np.ndarray | None = None,
        vs_2d: np.ndarray | None = None,
        rho_2d: np.ndarray | None = None,
    ) -> HybridResult:
        """Execute one forward simulation and return seismograms.

        Parameters
        ----------
        receivers_x, receivers_z : array_like
            Receiver positions (m).
        vp_2d, vs_2d, rho_2d : ndarray, shape (nz, nx), optional
            If provided, override the 2D material model for this run.
            Otherwise the default (laterally homogeneous from FK model) is used.
        """
        config = self.config
        receivers_x = np.atleast_1d(np.asarray(receivers_x, dtype=float))
        nrcv = receivers_x.size
        if receivers_z is None:
            receivers_z = np.zeros(nrcv)
        receivers_z = np.atleast_1d(np.asarray(receivers_z, dtype=float))

        if vp_2d is None and vs_2d is None and rho_2d is None:
            solver = self._solver
            solver.reset()
        else:
            vp = vp_2d if vp_2d is not None else self._vp_2d
            vs = vs_2d if vs_2d is not None else self._vs_2d
            rho = rho_2d if rho_2d is not None else self._rho_2d
            rho_ext, lam_ext, mu_ext = build_material_arrays(config, vp, vs, rho)
            solver = FDDisplacementSolver(config, rho_ext, lam_ext, mu_ext)

        seis_ux = np.zeros((nrcv, config.nt), dtype=np.float32)
        seis_uz = np.zeros((nrcv, config.nt), dtype=np.float32)
        rcv_ix, rcv_iz = _receiver_indices(receivers_x, receivers_z, config)

        bwf = self._bwf
        _init_coupling_nodes(solver, bwf, config)
        _dispatch_timeloop(
            solver, config, bwf,
            rcv_ix, rcv_iz,
            seis_ux, seis_uz,
            config.nt,
        )
        self._compiled = True

        if self.fortran_seisx_convention and self.phi != 0.0:
            seis_ux *= np.float32(np.cos(self.phi))

        return HybridResult(
            t=np.arange(config.nt) * config.dt,
            ux=seis_ux,
            uz=seis_uz,
            receivers_x=receivers_x,
            receivers_z=receivers_z,
            config=config,
        )
