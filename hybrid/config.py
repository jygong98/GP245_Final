"""Configuration dataclass for the hybrid FD-FK simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HybridConfig:
    """Parameters governing the hybrid FD-FK simulation grid.

    Attributes
    ----------
    nx, nz : int
        Number of interior grid points in x and z.
    dx, dz : float
        Grid spacing (m).
    x0, z0 : float
        Origin of the computational domain (m).
    dt : float
        Time step (s).
    nt : int
        Total number of time steps.
    norder : int
        **Half-order** of the spatial FD operator: the stencil uses
        ``norder`` grid points on each side of the centre, giving
        ``2 * norder``-th order spatial accuracy.  E.g. ``norder=8``
        → 16th-order FD.  ``FD_model.dat`` stores the full order
        (``fd_order``); pass ``fd_order // 2`` here.
        Default 3 (6th-order accuracy).
    npml : int
        Number of PML grid points on each absorbing boundary.
    kappa_max : float
        Maximum CFS-PML coordinate stretching factor (>= 1.0).
        1.0 disables stretching; values 1-7 improve evanescent wave
        absorption. Default 1.0 (backward-compatible with Sochacki).
    alpha_max : float
        Maximum CFS frequency shift (rad/s). Typically set to pi * f0
        where f0 is the dominant frequency. Improves low-frequency and
        grazing-incidence absorption. Default 0.0 (no shift).
    cpml_npower : int
        Polynomial order for the PML damping profile. Default 2.
    """

    nx: int
    nz: int
    dx: float
    dz: float
    x0: float = 0.0
    z0: float = 0.0
    dt: float = 0.01
    nt: int = 4501
    norder: int = 3
    npml: int = 20
    kappa_max: float = 1.0
    alpha_max: float = 0.0
    cpml_npower: int = 2

    @property
    def nbd(self) -> int:
        """Coupling-zone width in grid points (``2 * norder - 1``).

        The FD stencil at a coupling-zone boundary reaches ``norder``
        points into both the TF and SF sides; the overlapping centre
        row is shared, so the total strip width is ``2*norder - 1``.
        Must satisfy ``nbd <= npml`` to avoid PML injection overflow.
        """
        return 2 * self.norder - 1

    @property
    def xn(self) -> float:
        """Right edge of the domain (m)."""
        return self.x0 + (self.nx - 1) * self.dx

    @property
    def zn(self) -> float:
        """Bottom edge of the domain (m)."""
        return self.z0 + (self.nz - 1) * self.dz

    @property
    def x_coords(self) -> np.ndarray:
        """X-coordinates of all interior grid columns (m)."""
        return self.x0 + np.arange(self.nx) * self.dx

    @property
    def z_coords(self) -> np.ndarray:
        """Z-coordinates of all interior grid rows (m)."""
        return self.z0 + np.arange(self.nz) * self.dz

    @property
    def tmax(self) -> float:
        """Total simulation time (s)."""
        return (self.nt - 1) * self.dt
