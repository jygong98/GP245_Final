"""Wavefield injection (TF/SF coupling) for the hybrid FD-FK method.

Implements the total-field / scattered-field (TF/SF) decomposition at the
coupling zone between the regular domain and the PML absorbing layers.

The key idea (FDFK2D paper, eqs. 7a-7b):
  - In the regular domain, the FD evolves the TOTAL wavefield u_c.
  - In the PML domain, the FD evolves the SCATTERED wavefield u_s = u_c - u0.
  - The FD stencil at boundary nodes crosses the TF/SF interface.

Correct implementation requires TWO separate FD passes (matching Fortran):
  Pass 1 (regular domain): Add u0 to PML nodes so regular-domain stencils
    see total field everywhere.
  Pass 2 (PML domain): Subtract u0 from regular-domain coupling nodes so
    PML stencils see scattered field everywhere.

This module provides:
  - inject_for_regular_pass(): set up for Pass 1
  - restore_after_regular_pass(): clean up after Pass 1
  - inject_for_pml_pass(): set up for Pass 2
  - restore_after_pml_pass(): clean up after Pass 2
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .config import HybridConfig
from .background import BoundaryWavefield


def inject_for_regular_pass(solver, bwf: BoundaryWavefield,
                            config: HybridConfig, it: int):
    """Add u0 to PML coupling-zone nodes (cur and old time levels).

    After this call, PML nodes hold total field so that regular-domain
    stencils see consistent total field everywhere.

    At step 0, it_old=-1. We approximate bwf(t=-dt) with bwf(t=0) so
    that u_old at PML also holds consistent total field for the FD stencil.
    """
    nbd = config.nbd
    npml = config.npml
    nx, nz = config.nx, config.nz
    nt_bwf = bwf.ux_left.shape[-1]

    if 0 <= it < nt_bwf:
        _add_pml_sides(solver.ux_cur, solver.uz_cur, bwf, nbd, npml, nx, nz, it)
    it_old = max(it - 1, 0)
    if it_old < nt_bwf:
        _add_pml_sides(solver.ux_old, solver.uz_old, bwf, nbd, npml, nx, nz, it_old)


def restore_after_regular_pass(solver, bwf: BoundaryWavefield,
                               config: HybridConfig, it: int):
    """Subtract u0 from PML nodes to restore scattered field.

    Also subtract u0^{n+1} from ux_new in PML (the newly computed total
    field needs conversion to scattered before PML damping).
    """
    nbd = config.nbd
    npml = config.npml
    nx, nz = config.nx, config.nz
    nt_bwf = bwf.ux_left.shape[-1]

    # Restore u_cur and u_old
    if 0 <= it < nt_bwf:
        _sub_pml_sides(solver.ux_cur, solver.uz_cur, bwf, nbd, npml, nx, nz, it)
    it_old = max(it - 1, 0)
    if it_old < nt_bwf:
        _sub_pml_sides(solver.ux_old, solver.uz_old, bwf, nbd, npml, nx, nz, it_old)

    # Convert u_new in PML to scattered (subtract u0^{n+1})
    it_new = it + 1
    if 0 <= it_new < nt_bwf:
        _sub_pml_sides(solver.ux_new, solver.uz_new, bwf, nbd, npml, nx, nz, it_new)


def inject_for_pml_pass(solver, bwf: BoundaryWavefield,
                        config: HybridConfig, it: int):
    """Subtract u0 from regular coupling-zone nodes (cur and old).

    After this call, regular-domain coupling nodes hold scattered field
    so that PML stencils see consistent scattered field everywhere.
    """
    nbd = config.nbd
    norder = config.norder
    npml = config.npml
    nx, nz = config.nx, config.nz
    nt_bwf = bwf.ux_left.shape[-1]

    if 0 <= it < nt_bwf:
        _sub_reg_sides(solver.ux_cur, solver.uz_cur, bwf,
                       nbd, norder, npml, nx, nz, it)
    it_old = max(it - 1, 0)
    if it_old < nt_bwf:
        _sub_reg_sides(solver.ux_old, solver.uz_old, bwf,
                       nbd, norder, npml, nx, nz, it_old)


def restore_after_pml_pass(solver, bwf: BoundaryWavefield,
                           config: HybridConfig, it: int):
    """Add u0 back to regular coupling-zone nodes (cur and old).

    Restores regular-domain nodes to total field after PML pass.
    """
    nbd = config.nbd
    norder = config.norder
    npml = config.npml
    nx, nz = config.nx, config.nz
    nt_bwf = bwf.ux_left.shape[-1]

    if 0 <= it < nt_bwf:
        _add_reg_sides(solver.ux_cur, solver.uz_cur, bwf,
                       nbd, norder, npml, nx, nz, it)
    it_old = max(it - 1, 0)
    if it_old < nt_bwf:
        _add_reg_sides(solver.ux_old, solver.uz_old, bwf,
                       nbd, norder, npml, nx, nz, it_old)


# ===================================================================
# PML-side add/subtract (for regular-domain pass)
# ===================================================================

def _add_pml_sides(ux, uz, bwf, nbd, npml, nx, nz, it):
    """Add u0 to PML coupling-zone nodes on all three boundaries."""
    _op_left_pml(ux, uz, bwf.ux_left[:, :nbd, it], bwf.uz_left[:, :nbd, it],
                 nbd, npml, nz, add=True)
    _op_right_pml(ux, uz, bwf.ux_right[:, nbd:, it], bwf.uz_right[:, nbd:, it],
                  nbd, npml, nx, nz, add=True)
    _op_bottom_pml(ux, uz, bwf.ux_bottom[:, nbd:, it], bwf.uz_bottom[:, nbd:, it],
                   nbd, npml, nx, nz, add=True)


def _sub_pml_sides(ux, uz, bwf, nbd, npml, nx, nz, it):
    """Subtract u0 from PML coupling-zone nodes."""
    _op_left_pml(ux, uz, bwf.ux_left[:, :nbd, it], bwf.uz_left[:, :nbd, it],
                 nbd, npml, nz, add=False)
    _op_right_pml(ux, uz, bwf.ux_right[:, nbd:, it], bwf.uz_right[:, nbd:, it],
                  nbd, npml, nx, nz, add=False)
    _op_bottom_pml(ux, uz, bwf.ux_bottom[:, nbd:, it], bwf.uz_bottom[:, nbd:, it],
                   nbd, npml, nx, nz, add=False)


# ===================================================================
# Regular-side add/subtract (for PML pass)
# ===================================================================

def _sub_reg_sides(ux, uz, bwf, nbd, norder, npml, nx, nz, it):
    """Subtract u0 from regular coupling-zone nodes accessed by PML stencils.

    Left/right injection covers only the top nz-nbd rows; the bottom nbd
    rows are handled exclusively by the bottom injection to avoid corner overlap.
    Bottom injection covers only regular-domain x-columns (matching Fortran).
    """
    nz_top = nz - nbd
    _op_left_reg(ux, uz, bwf.ux_left[:nz_top, nbd:nbd+norder, it],
                 bwf.uz_left[:nz_top, nbd:nbd+norder, it],
                 nbd, norder, npml, nz, add=False)
    _op_right_reg(ux, uz, bwf.ux_right[:nz_top, nbd-norder:nbd, it],
                  bwf.uz_right[:nz_top, nbd-norder:nbd, it],
                  nbd, norder, npml, nx, nz, add=False)
    _op_bottom_reg(ux, uz, bwf.ux_bottom[nbd:nbd+nx, :nbd, it],
                   bwf.uz_bottom[nbd:nbd+nx, :nbd, it],
                   nbd, norder, npml, nx, nz, add=False)


def _add_reg_sides(ux, uz, bwf, nbd, norder, npml, nx, nz, it):
    """Add u0 back to regular coupling-zone nodes.

    Left/right injection covers only the top nz-nbd rows; the bottom nbd
    rows are handled exclusively by the bottom injection to avoid corner overlap.
    Bottom injection covers only regular-domain x-columns (matching Fortran).
    """
    nz_top = nz - nbd
    _op_left_reg(ux, uz, bwf.ux_left[:nz_top, nbd:nbd+norder, it],
                 bwf.uz_left[:nz_top, nbd:nbd+norder, it],
                 nbd, norder, npml, nz, add=True)
    _op_right_reg(ux, uz, bwf.ux_right[:nz_top, nbd-norder:nbd, it],
                  bwf.uz_right[:nz_top, nbd-norder:nbd, it],
                  nbd, norder, npml, nx, nz, add=True)
    _op_bottom_reg(ux, uz, bwf.ux_bottom[nbd:nbd+nx, :nbd, it],
                   bwf.uz_bottom[nbd:nbd+nx, :nbd, it],
                   nbd, norder, npml, nx, nz, add=True)


# ===================================================================
# Low-level kernels for PML-side nodes
# ===================================================================

@njit(cache=True)
def _op_left_pml(ux, uz, ux0, uz0, nbd, npml, nz, add):
    """Add/subtract u0 at left PML coupling nodes.

    Background columns 0..nbd-1 map to ix_ext = npml-nbd..npml-1.
    When nbd > npml, only the rightmost npml columns are injected
    (columns with non-negative ix_ext).
    """
    sign = 1.0 if add else -1.0
    ic_start = max(0, nbd - npml)
    for iz in range(nz):
        for ic in range(ic_start, nbd):
            ix_ext = npml - nbd + ic
            ux[iz, ix_ext] += sign * ux0[iz, ic]
            uz[iz, ix_ext] += sign * uz0[iz, ic]


@njit(cache=True)
def _op_right_pml(ux, uz, ux0, uz0, nbd, npml, nx, nz, add):
    """Add/subtract u0 at right PML coupling nodes.

    Background columns 0..nbd-1 (PML side of right bwf) map to
    ix_ext = npml+nx .. npml+nx+nbd-1.
    When nbd > npml, only the first npml columns are injected
    (columns within the array bounds).
    """
    sign = 1.0 if add else -1.0
    nx_tot = nx + 2 * npml
    ic_end = min(nbd, npml)
    for iz in range(nz):
        for ic in range(ic_end):
            ix_ext = npml + nx + ic
            ux[iz, ix_ext] += sign * ux0[iz, ic]
            uz[iz, ix_ext] += sign * uz0[iz, ic]


@njit(cache=True)
def _op_bottom_pml(ux, uz, ux0, uz0, nbd, npml, nx, nz, add):
    """Add/subtract u0 at bottom PML coupling nodes.

    Background rows 0..nbd-1 (PML side of bottom bwf) map to
    iz = nz..nz+nbd-1.
    The bottom bwf extends nbd columns into left/right PML (corners),
    so first axis covers nx+2*nbd columns starting from ix_reg = -nbd.
    When nbd > npml, only the first npml rows (within the grid) are used.
    """
    sign = 1.0 if add else -1.0
    nx_ext_bot = ux0.shape[0]  # nx + 2*nbd
    nx_tot = ux.shape[1]
    nz_tot = ux.shape[0]
    ir_end = min(nbd, npml)
    for i in range(nx_ext_bot):
        ix_ext = npml - nbd + i
        if ix_ext < 0 or ix_ext >= nx_tot:
            continue
        for ir in range(ir_end):
            iz = nz + ir
            ux[iz, ix_ext] += sign * ux0[i, ir]
            uz[iz, ix_ext] += sign * uz0[i, ir]


# ===================================================================
# Low-level kernels for regular-side nodes
# ===================================================================

@njit(cache=True)
def _op_left_reg(ux, uz, ux0, uz0, nbd, norder, npml, nz, add):
    """Add/subtract u0 at regular nodes accessed by left-PML stencils.

    Regular-side: first norder columns of regular domain (ix_reg = 0..norder-1).
    Only the top nz-nbd rows are handled here; the bottom nbd rows are
    handled by _op_bottom_reg to avoid double injection at corners.
    """
    sign = 1.0 if add else -1.0
    nz_top = nz - nbd
    for iz in range(nz_top):
        for ic in range(norder):
            ix_ext = npml + ic
            ux[iz, ix_ext] += sign * ux0[iz, ic]
            uz[iz, ix_ext] += sign * uz0[iz, ic]


@njit(cache=True)
def _op_right_reg(ux, uz, ux0, uz0, nbd, norder, npml, nx, nz, add):
    """Add/subtract u0 at regular nodes accessed by right-PML stencils.

    Regular-side: last norder columns (ix_reg = nx-norder..nx-1).
    Only the top nz-nbd rows are handled here; the bottom nbd rows are
    handled by _op_bottom_reg to avoid double injection at corners.
    """
    sign = 1.0 if add else -1.0
    nz_top = nz - nbd
    for iz in range(nz_top):
        for ic in range(norder):
            ix_ext = npml + nx - norder + ic
            ux[iz, ix_ext] += sign * ux0[iz, ic]
            uz[iz, ix_ext] += sign * uz0[iz, ic]


@njit(cache=True)
def _op_bottom_reg(ux, uz, ux0, uz0, nbd, norder, npml, nx, nz, add):
    """Add/subtract u0 at regular nodes accessed by bottom-PML stencils.

    Regular-side: bottom nbd rows (iz = nz-nbd..nz-1), covering only the
    regular-domain x-columns (ix = npml..npml+nx-1). Matches the Fortran
    uxb_reg(nz-nbd+1:nz, 1:nx) allocation. PML x-columns are NOT modified
    here; they are handled by the PML-side bottom injection (_op_bottom_pml).
    """
    sign = 1.0 if add else -1.0
    nx_reg = ux0.shape[0]  # should be nx (regular domain only)
    for i in range(nx_reg):
        ix_ext = npml + i
        for ir in range(nbd):
            iz = nz - nbd + ir
            ux[iz, ix_ext] += sign * ux0[i, ir]
            uz[iz, ix_ext] += sign * uz0[i, ir]
