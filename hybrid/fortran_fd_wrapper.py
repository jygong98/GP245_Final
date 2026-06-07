"""Ctypes wrapper for Fortran libHybrid.so::updatedispl subroutine.

This module provides a thin Python interface to the EXACT Fortran FD kernel
used in the FDFK2D code (Liu et al., 2025), enabling bit-for-bit comparison
between the Python FD implementation and the original Fortran library.

Requirements:
    - Linux x86-64 (ELF shared libraries)
    - libHybrid.so, libPML.so compiled with gfortran (or Intel)
    - libSubmain.so (custom build from Submain.f90 for diff* symbols)
    - libgfortran, libgomp available in LD_LIBRARY_PATH

Symbol: updatedispl_ (standard Fortran name-mangling with trailing underscore)
Convention: ALL arguments passed by reference (Fortran default)
"""

from __future__ import annotations

import ctypes
import os
import platform
from ctypes import (
    POINTER,
    byref,
    c_double,
    c_float,
    c_int,
    c_int32,
)
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Fortran array helper
# ---------------------------------------------------------------------------

def _f_ptr(arr: np.ndarray, dtype=np.float32) -> ctypes.c_void_p:
    """Return a ctypes pointer to the data of a Fortran-contiguous array.

    Validates dtype and contiguity before returning the pointer.
    """
    if arr.dtype != dtype:
        raise TypeError(
            f"Expected dtype {dtype}, got {arr.dtype}. "
            f"Array must match the Fortran declaration exactly."
        )
    if not arr.flags['F_CONTIGUOUS']:
        raise ValueError(
            "Array must be Fortran-contiguous (column-major, order='F'). "
            "Use np.asfortranarray() to convert."
        )
    return arr.ctypes.data_as(ctypes.c_void_p)


def _f_ptr64(arr: np.ndarray) -> ctypes.c_void_p:
    """Return ctypes pointer for a float64 Fortran-contiguous array."""
    return _f_ptr(arr, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class FortranHybridFD:
    """Wrapper around libHybrid.so::updatedispl for calling the Fortran FD kernel.

    The updatedispl subroutine performs ONE time step of the 2D elastic
    displacement-based finite-difference computation with:
      - High-order spatial FD operators (collocated grid)
      - C-PML absorbing boundaries (left, right, bottom; optionally top)
      - FK background wavefield injection at domain boundaries (Hybrid method)

    Parameters (68 total):
        Scalars (15):
            norder, nx, nz, npml, izt, is_PML_top, npower, nbd, it, nt,
            dx, dz, dt, vpml, R0
        FK background wavefields (12 arrays):
            uxl_reg, uzl_reg, uxr_reg, uzr_reg, uxb_reg, uzb_reg,
            uxl_pml, uzl_pml, uxr_pml, uzr_pml, uxb_pml, uzb_pml
        Material properties (1 array):
            c
        FD coefficients (2 arrays):
            coef0, coef1
        Displacement fields (2 arrays, MODIFIED in-place):
            ux, uz
        PML auxiliary fields (12 2D arrays, MODIFIED in-place):
            bpxlx, bpzlx, bpxrx, bpzrx, bpxtx, bpxtz,
            bpztx, bpztz, bpxbx, bpxbz, bpzbx, bpzbz
        PML memory variables (24 3D arrays, MODIFIED in-place):
            bux1l..buz3l, bux1r..buz3r, bux1b..buz3b, bux1t..buz3t

    Usage on Sherlock:
        >>> wrapper = FortranHybridFD("/path/to/libs")
        >>> wrapper.step(norder, nx, nz, ..., ux, uz, ...)
        # ux, uz are modified in-place
    """

    # Relative library load order matters: dependencies must be loaded first
    _LIB_LOAD_ORDER = ("libSubmain.so", "libPML.so", "libHybrid.so")

    def __init__(self, lib_dir: str, compiler: str = "gnu"):
        """Load the Fortran shared libraries.

        Parameters
        ----------
        lib_dir : str
            Directory containing all required .so files.
            Expected: libSubmain.so, libPML.so, libHybrid.so
        compiler : str
            Compiler flavor: 'gnu', 'intel', or 'pgi'.
            Affects expected library subdirectory if lib_dir is the project root.
        """
        if platform.system() != "Linux":
            raise RuntimeError(
                f"FortranHybridFD requires Linux (ELF shared libraries). "
                f"Current platform: {platform.system()}. "
                f"Run this on Sherlock or a Linux machine."
            )

        lib_path = Path(lib_dir)
        if not lib_path.is_dir():
            raise FileNotFoundError(f"Library directory not found: {lib_dir}")

        self._libs = {}
        for lib_name in self._LIB_LOAD_ORDER:
            so_path = lib_path / lib_name
            if not so_path.exists():
                raise FileNotFoundError(
                    f"Required library not found: {so_path}\n"
                    f"See doc/ctypes_libhybrid_wrapper.md for build instructions."
                )
            self._libs[lib_name] = ctypes.CDLL(
                str(so_path), mode=ctypes.RTLD_GLOBAL
            )

        self._updatedispl = self._libs["libHybrid.so"].updatedispl_
        self._updatedispl.restype = None

    def step(
        self,
        # --- Scalars ---
        norder: int,
        nx: int,
        nz: int,
        npml: int,
        izt: int,
        is_pml_top: bool,
        npower: int,
        nbd: int,
        it: int,
        nt: int,
        dx: float,
        dz: float,
        dt: float,
        vpml: float,
        R0: float,
        # --- FK background wavefields (float32, F-order) ---
        uxl_reg: np.ndarray,
        uzl_reg: np.ndarray,
        uxr_reg: np.ndarray,
        uzr_reg: np.ndarray,
        uxb_reg: np.ndarray,
        uzb_reg: np.ndarray,
        uxl_pml: np.ndarray,
        uzl_pml: np.ndarray,
        uxr_pml: np.ndarray,
        uzr_pml: np.ndarray,
        uxb_pml: np.ndarray,
        uzb_pml: np.ndarray,
        # --- Material model (float32, F-order) ---
        c: np.ndarray,
        # --- FD coefficients (float64, F-order) ---
        coef0: np.ndarray,
        coef1: np.ndarray,
        # --- Displacement fields (float32, F-order, MODIFIED) ---
        ux: np.ndarray,
        uz: np.ndarray,
        # --- PML 2D auxiliary (float32, F-order, MODIFIED) ---
        bpxlx: np.ndarray,
        bpzlx: np.ndarray,
        bpxrx: np.ndarray,
        bpzrx: np.ndarray,
        bpxtx: np.ndarray,
        bpxtz: np.ndarray,
        bpztx: np.ndarray,
        bpztz: np.ndarray,
        bpxbx: np.ndarray,
        bpxbz: np.ndarray,
        bpzbx: np.ndarray,
        bpzbz: np.ndarray,
        # --- PML 3D memory variables (float32, F-order, MODIFIED) ---
        bux1l: np.ndarray,
        bux2l: np.ndarray,
        bux3l: np.ndarray,
        buz1l: np.ndarray,
        buz2l: np.ndarray,
        buz3l: np.ndarray,
        bux1r: np.ndarray,
        bux2r: np.ndarray,
        bux3r: np.ndarray,
        buz1r: np.ndarray,
        buz2r: np.ndarray,
        buz3r: np.ndarray,
        bux1b: np.ndarray,
        bux2b: np.ndarray,
        bux3b: np.ndarray,
        buz1b: np.ndarray,
        buz2b: np.ndarray,
        buz3b: np.ndarray,
        bux1t: np.ndarray,
        bux2t: np.ndarray,
        bux3t: np.ndarray,
        buz1t: np.ndarray,
        buz2t: np.ndarray,
        buz3t: np.ndarray,
    ) -> None:
        """Execute one FD time step via the Fortran updatedispl subroutine.

        All array arguments must be:
          - Fortran-contiguous (order='F')
          - Correct dtype (float32 for real, float64 for real*8)
          - Correct shape matching the Fortran allocation

        Displacement arrays (ux, uz) and PML auxiliary arrays are MODIFIED
        in-place by the Fortran subroutine.

        Array dimension reference (Fortran indexing):
            ux, uz: (2, izt:nz+npml, 1-npml:nx+npml)
                → numpy shape (2, nz+npml-izt+1, nx+2*npml)
            c: (3, izt:nz+npml, 1-npml:nx+npml)
                → numpy shape (3, nz+npml-izt+1, nx+2*npml)
            coef0, coef1: (norder, norder) float64
            uxl_reg: (nz-nbd, nbd, nt+1) float32
            uzl_reg: same
            uxr_reg: (nz-nbd, nbd, nt+1) float32
            uzr_reg: same
            uxb_reg: (nbd, nx, nt+1) float32
            uzb_reg: same
            uxl_pml: (nz, nbd, nt+1) float32
            uzl_pml: same
            uxr_pml: (nz, nbd, nt+1) float32
            uzr_pml: same
            uxb_pml: (nbd, nx+2*nbd, nt+1) float32
            uzb_pml: same
            bpxlx, bpzlx: (nz, npml) float32
            bpxrx, bpzrx: (nz, npml) float32
            bpxtx, bpxtz, bpztx, bpztz: (npml, nx+2*npml) float32
            bpxbx, bpxbz, bpzbx, bpzbz: (npml, nx+2*npml) float32
            bux1l..buz3l: (2, nz, npml) float32
            bux1r..buz3r: (2, nz, npml) float32
            bux1b..buz3b: (2, npml, nx+2*npml) float32
            bux1t..buz3t: (2, npml, nx+2*npml) float32
        """
        # --- Pack scalar arguments as ctypes by-reference ---
        # Fortran passes everything by reference
        i_norder = c_int(norder)
        i_nx = c_int(nx)
        i_nz = c_int(nz)
        i_npml = c_int(npml)
        i_izt = c_int(izt)
        # gfortran logical is 4-byte int: .TRUE.=1, .FALSE.=0
        i_is_pml_top = c_int32(1 if is_pml_top else 0)
        i_npower = c_int(npower)
        i_nbd = c_int(nbd)
        i_it = c_int(it)
        i_nt = c_int(nt)
        f_dx = c_float(dx)
        f_dz = c_float(dz)
        d_dt = c_double(dt)
        f_vpml = c_float(vpml)
        d_R0 = c_double(R0)

        # --- Validate and get array pointers ---
        # FK background wavefields (read-only, float32)
        p_uxl_reg = _f_ptr(uxl_reg)
        p_uzl_reg = _f_ptr(uzl_reg)
        p_uxr_reg = _f_ptr(uxr_reg)
        p_uzr_reg = _f_ptr(uzr_reg)
        p_uxb_reg = _f_ptr(uxb_reg)
        p_uzb_reg = _f_ptr(uzb_reg)
        p_uxl_pml = _f_ptr(uxl_pml)
        p_uzl_pml = _f_ptr(uzl_pml)
        p_uxr_pml = _f_ptr(uxr_pml)
        p_uzr_pml = _f_ptr(uzr_pml)
        p_uxb_pml = _f_ptr(uxb_pml)
        p_uzb_pml = _f_ptr(uzb_pml)

        # Material model (read-only, float32)
        p_c = _f_ptr(c)

        # FD coefficients (read-only, float64)
        p_coef0 = _f_ptr64(coef0)
        p_coef1 = _f_ptr64(coef1)

        # Displacement fields (read-write, float32)
        p_ux = _f_ptr(ux)
        p_uz = _f_ptr(uz)

        # PML 2D auxiliary (read-write, float32)
        p_bpxlx = _f_ptr(bpxlx)
        p_bpzlx = _f_ptr(bpzlx)
        p_bpxrx = _f_ptr(bpxrx)
        p_bpzrx = _f_ptr(bpzrx)
        p_bpxtx = _f_ptr(bpxtx)
        p_bpxtz = _f_ptr(bpxtz)
        p_bpztx = _f_ptr(bpztx)
        p_bpztz = _f_ptr(bpztz)
        p_bpxbx = _f_ptr(bpxbx)
        p_bpxbz = _f_ptr(bpxbz)
        p_bpzbx = _f_ptr(bpzbx)
        p_bpzbz = _f_ptr(bpzbz)

        # PML 3D memory variables (read-write, float32)
        p_bux1l = _f_ptr(bux1l)
        p_bux2l = _f_ptr(bux2l)
        p_bux3l = _f_ptr(bux3l)
        p_buz1l = _f_ptr(buz1l)
        p_buz2l = _f_ptr(buz2l)
        p_buz3l = _f_ptr(buz3l)
        p_bux1r = _f_ptr(bux1r)
        p_bux2r = _f_ptr(bux2r)
        p_bux3r = _f_ptr(bux3r)
        p_buz1r = _f_ptr(buz1r)
        p_buz2r = _f_ptr(buz2r)
        p_buz3r = _f_ptr(buz3r)
        p_bux1b = _f_ptr(bux1b)
        p_bux2b = _f_ptr(bux2b)
        p_bux3b = _f_ptr(bux3b)
        p_buz1b = _f_ptr(buz1b)
        p_buz2b = _f_ptr(buz2b)
        p_buz3b = _f_ptr(buz3b)
        p_bux1t = _f_ptr(bux1t)
        p_bux2t = _f_ptr(bux2t)
        p_bux3t = _f_ptr(bux3t)
        p_buz1t = _f_ptr(buz1t)
        p_buz2t = _f_ptr(buz2t)
        p_buz3t = _f_ptr(buz3t)

        # --- Call Fortran subroutine ---
        # All arguments by reference (Fortran calling convention)
        self._updatedispl(
            byref(i_norder), byref(i_nx), byref(i_nz), byref(i_npml),
            byref(i_izt), byref(i_is_pml_top), byref(i_npower), byref(i_nbd),
            byref(i_it), byref(i_nt),
            byref(f_dx), byref(f_dz), byref(d_dt), byref(f_vpml), byref(d_R0),
            # FK background wavefields
            p_uxl_reg, p_uzl_reg, p_uxr_reg, p_uzr_reg,
            p_uxb_reg, p_uzb_reg,
            p_uxl_pml, p_uzl_pml, p_uxr_pml, p_uzr_pml,
            p_uxb_pml, p_uzb_pml,
            # Material, FD coefficients
            p_c, p_coef0, p_coef1,
            # Displacement fields
            p_ux, p_uz,
            # PML 2D auxiliary
            p_bpxlx, p_bpzlx, p_bpxrx, p_bpzrx,
            p_bpxtx, p_bpxtz, p_bpztx, p_bpztz,
            p_bpxbx, p_bpxbz, p_bpzbx, p_bpzbz,
            # PML 3D memory variables
            p_bux1l, p_bux2l, p_bux3l, p_buz1l, p_buz2l, p_buz3l,
            p_bux1r, p_bux2r, p_bux3r, p_buz1r, p_buz2r, p_buz3r,
            p_bux1b, p_bux2b, p_bux3b, p_buz1b, p_buz2b, p_buz3b,
            p_bux1t, p_bux2t, p_bux3t, p_buz1t, p_buz2t, p_buz3t,
        )


# ---------------------------------------------------------------------------
# Helper: allocate arrays matching Fortran dimensions
# ---------------------------------------------------------------------------

def allocate_fortran_arrays(
    norder: int, nx: int, nz: int, npml: int, izt: int, nt: int
) -> dict:
    """Allocate zero-initialized arrays matching Fortran dimensions.

    Returns a dict of array names → numpy arrays (Fortran-contiguous).
    Use for initializing PML auxiliary and displacement fields.

    Parameters
    ----------
    norder : int
        FD half-order (e.g. 8 for 16th-order).
    nx, nz : int
        Interior grid dimensions.
    npml : int
        Number of PML layers.
    izt : int
        Starting z-index (1 if no top PML, 1-npml if top PML).
    nt : int
        Total time steps.
    """
    nbd = 2 * norder - 1
    nz_ext = nz + npml - izt + 1  # total z-extent including PML
    nx_ext = nx + 2 * npml         # total x-extent including PML

    arrays = {}

    # Displacement fields: shape (2, nz_ext, nx_ext)
    arrays['ux'] = np.zeros((2, nz_ext, nx_ext), dtype=np.float32, order='F')
    arrays['uz'] = np.zeros((2, nz_ext, nx_ext), dtype=np.float32, order='F')

    # Material model: shape (3, nz_ext, nx_ext)
    # (must be filled externally with actual model)
    arrays['c'] = np.zeros((3, nz_ext, nx_ext), dtype=np.float32, order='F')

    # FD coefficients
    arrays['coef0'] = np.zeros((norder, norder), dtype=np.float64, order='F')
    arrays['coef1'] = np.zeros((norder, norder), dtype=np.float64, order='F')

    # FK background wavefields (all time steps)
    arrays['uxl_reg'] = np.zeros((nz - nbd, nbd, nt + 1), dtype=np.float32, order='F')
    arrays['uzl_reg'] = np.zeros((nz - nbd, nbd, nt + 1), dtype=np.float32, order='F')
    arrays['uxr_reg'] = np.zeros((nz - nbd, nbd, nt + 1), dtype=np.float32, order='F')
    arrays['uzr_reg'] = np.zeros((nz - nbd, nbd, nt + 1), dtype=np.float32, order='F')
    arrays['uxb_reg'] = np.zeros((nbd, nx, nt + 1), dtype=np.float32, order='F')
    arrays['uzb_reg'] = np.zeros((nbd, nx, nt + 1), dtype=np.float32, order='F')

    arrays['uxl_pml'] = np.zeros((nz, nbd, nt + 1), dtype=np.float32, order='F')
    arrays['uzl_pml'] = np.zeros((nz, nbd, nt + 1), dtype=np.float32, order='F')
    arrays['uxr_pml'] = np.zeros((nz, nbd, nt + 1), dtype=np.float32, order='F')
    arrays['uzr_pml'] = np.zeros((nz, nbd, nt + 1), dtype=np.float32, order='F')
    arrays['uxb_pml'] = np.zeros((nbd, nx + 2 * nbd, nt + 1), dtype=np.float32, order='F')
    arrays['uzb_pml'] = np.zeros((nbd, nx + 2 * nbd, nt + 1), dtype=np.float32, order='F')

    # PML 2D auxiliary fields
    arrays['bpxlx'] = np.zeros((nz, npml), dtype=np.float32, order='F')
    arrays['bpzlx'] = np.zeros((nz, npml), dtype=np.float32, order='F')
    arrays['bpxrx'] = np.zeros((nz, npml), dtype=np.float32, order='F')
    arrays['bpzrx'] = np.zeros((nz, npml), dtype=np.float32, order='F')

    arrays['bpxtx'] = np.zeros((npml, nx_ext), dtype=np.float32, order='F')
    arrays['bpxtz'] = np.zeros((npml, nx_ext), dtype=np.float32, order='F')
    arrays['bpztx'] = np.zeros((npml, nx_ext), dtype=np.float32, order='F')
    arrays['bpztz'] = np.zeros((npml, nx_ext), dtype=np.float32, order='F')

    arrays['bpxbx'] = np.zeros((npml, nx_ext), dtype=np.float32, order='F')
    arrays['bpxbz'] = np.zeros((npml, nx_ext), dtype=np.float32, order='F')
    arrays['bpzbx'] = np.zeros((npml, nx_ext), dtype=np.float32, order='F')
    arrays['bpzbz'] = np.zeros((npml, nx_ext), dtype=np.float32, order='F')

    # PML 3D memory variables: left boundary
    arrays['bux1l'] = np.zeros((2, nz, npml), dtype=np.float32, order='F')
    arrays['bux2l'] = np.zeros((2, nz, npml), dtype=np.float32, order='F')
    arrays['bux3l'] = np.zeros((2, nz, npml), dtype=np.float32, order='F')
    arrays['buz1l'] = np.zeros((2, nz, npml), dtype=np.float32, order='F')
    arrays['buz2l'] = np.zeros((2, nz, npml), dtype=np.float32, order='F')
    arrays['buz3l'] = np.zeros((2, nz, npml), dtype=np.float32, order='F')

    # PML 3D memory variables: right boundary
    arrays['bux1r'] = np.zeros((2, nz, npml), dtype=np.float32, order='F')
    arrays['bux2r'] = np.zeros((2, nz, npml), dtype=np.float32, order='F')
    arrays['bux3r'] = np.zeros((2, nz, npml), dtype=np.float32, order='F')
    arrays['buz1r'] = np.zeros((2, nz, npml), dtype=np.float32, order='F')
    arrays['buz2r'] = np.zeros((2, nz, npml), dtype=np.float32, order='F')
    arrays['buz3r'] = np.zeros((2, nz, npml), dtype=np.float32, order='F')

    # PML 3D memory variables: bottom boundary
    arrays['bux1b'] = np.zeros((2, npml, nx_ext), dtype=np.float32, order='F')
    arrays['bux2b'] = np.zeros((2, npml, nx_ext), dtype=np.float32, order='F')
    arrays['bux3b'] = np.zeros((2, npml, nx_ext), dtype=np.float32, order='F')
    arrays['buz1b'] = np.zeros((2, npml, nx_ext), dtype=np.float32, order='F')
    arrays['buz2b'] = np.zeros((2, npml, nx_ext), dtype=np.float32, order='F')
    arrays['buz3b'] = np.zeros((2, npml, nx_ext), dtype=np.float32, order='F')

    # PML 3D memory variables: top boundary
    arrays['bux1t'] = np.zeros((2, npml, nx_ext), dtype=np.float32, order='F')
    arrays['bux2t'] = np.zeros((2, npml, nx_ext), dtype=np.float32, order='F')
    arrays['bux3t'] = np.zeros((2, npml, nx_ext), dtype=np.float32, order='F')
    arrays['buz1t'] = np.zeros((2, npml, nx_ext), dtype=np.float32, order='F')
    arrays['buz2t'] = np.zeros((2, npml, nx_ext), dtype=np.float32, order='F')
    arrays['buz3t'] = np.zeros((2, npml, nx_ext), dtype=np.float32, order='F')

    return arrays


def validate_array_shapes(
    norder: int, nx: int, nz: int, npml: int, izt: int, nt: int,
    **arrays
) -> list[str]:
    """Validate array shapes against expected Fortran dimensions.

    Returns list of error messages (empty if all shapes are correct).
    """
    nbd = 2 * norder - 1
    nz_ext = nz + npml - izt + 1
    nx_ext = nx + 2 * npml

    expected = {
        'ux': (2, nz_ext, nx_ext),
        'uz': (2, nz_ext, nx_ext),
        'c': (3, nz_ext, nx_ext),
        'coef0': (norder, norder),
        'coef1': (norder, norder),
        'uxl_reg': (nz - nbd, nbd, nt + 1),
        'uzl_reg': (nz - nbd, nbd, nt + 1),
        'uxr_reg': (nz - nbd, nbd, nt + 1),
        'uzr_reg': (nz - nbd, nbd, nt + 1),
        'uxb_reg': (nbd, nx, nt + 1),
        'uzb_reg': (nbd, nx, nt + 1),
        'uxl_pml': (nz, nbd, nt + 1),
        'uzl_pml': (nz, nbd, nt + 1),
        'uxr_pml': (nz, nbd, nt + 1),
        'uzr_pml': (nz, nbd, nt + 1),
        'uxb_pml': (nbd, nx + 2 * nbd, nt + 1),
        'uzb_pml': (nbd, nx + 2 * nbd, nt + 1),
        'bpxlx': (nz, npml),
        'bpzlx': (nz, npml),
        'bpxrx': (nz, npml),
        'bpzrx': (nz, npml),
        'bpxtx': (npml, nx_ext),
        'bpxtz': (npml, nx_ext),
        'bpztx': (npml, nx_ext),
        'bpztz': (npml, nx_ext),
        'bpxbx': (npml, nx_ext),
        'bpxbz': (npml, nx_ext),
        'bpzbx': (npml, nx_ext),
        'bpzbz': (npml, nx_ext),
        'bux1l': (2, nz, npml), 'bux2l': (2, nz, npml), 'bux3l': (2, nz, npml),
        'buz1l': (2, nz, npml), 'buz2l': (2, nz, npml), 'buz3l': (2, nz, npml),
        'bux1r': (2, nz, npml), 'bux2r': (2, nz, npml), 'bux3r': (2, nz, npml),
        'buz1r': (2, nz, npml), 'buz2r': (2, nz, npml), 'buz3r': (2, nz, npml),
        'bux1b': (2, npml, nx_ext), 'bux2b': (2, npml, nx_ext), 'bux3b': (2, npml, nx_ext),
        'buz1b': (2, npml, nx_ext), 'buz2b': (2, npml, nx_ext), 'buz3b': (2, npml, nx_ext),
        'bux1t': (2, npml, nx_ext), 'bux2t': (2, npml, nx_ext), 'bux3t': (2, npml, nx_ext),
        'buz1t': (2, npml, nx_ext), 'buz2t': (2, npml, nx_ext), 'buz3t': (2, npml, nx_ext),
    }

    errors = []
    for name, arr in arrays.items():
        if name in expected:
            if arr.shape != expected[name]:
                errors.append(
                    f"{name}: expected shape {expected[name]}, got {arr.shape}"
                )
            if not arr.flags['F_CONTIGUOUS']:
                errors.append(f"{name}: not Fortran-contiguous")
    return errors
