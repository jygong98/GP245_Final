"""Source definitions and standalone runner for the FD displacement solver.

Provides body-force and moment-tensor source classes used for standalone
FD validation (not part of the hybrid FDFK workflow), plus a time-marching
utility that drives the solver with an internal source.

Source classes
--------------
- ``PointBodyForceSource`` — point body force (dipole radiation)
- ``LineBodyForceSource`` — line body force spanning all x (dipole radiation)
- ``ExplosiveLineSource`` — isotropic explosion line (monopole radiation)
- ``ExplosivePointSource`` — isotropic explosion point (monopole radiation)

Functions
---------
- ``inject_source`` — dispatch source injection into solver fields
- ``run_standalone`` — complete time-march loop with source and receivers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .fd_displacement import FDDisplacementSolver


# ===================================================================
# Source classes
# ===================================================================

@dataclass
class PointBodyForceSource:
    """Explosive body-force source at a single grid location.

    Injects equally into ux and uz (dipole radiation pattern).

    Parameters
    ----------
    ix_ext : int
        X-index on the extended grid (including PML padding).
    iz : int
        Z-index on the extended grid.
    wavelet : ndarray, shape (nt,)
        Source time function amplitude at each time step.
    """
    ix_ext: int
    iz: int
    wavelet: np.ndarray


@dataclass
class LineBodyForceSource:
    """Body-force line source spanning all x-columns at a given depth.

    Injects the same body force at (iz, ix) for all ix in the regular
    domain.  Produces a dipole radiation pattern.

    Parameters
    ----------
    iz : int
        Z-index on the extended grid.
    wavelet : ndarray, shape (nt,)
        Source time function amplitude at each time step.
    """
    iz: int
    wavelet: np.ndarray


@dataclass
class ExplosiveLineSource:
    """Isotropic explosion line source (moment-tensor monopole).

    Equivalent to a line of explosion sources spanning all x-columns at
    depth ``iz``.  Unlike ``LineBodyForceSource`` (which injects equal body
    forces into ux and uz and produces a dipole radiation pattern), this
    source injects via the stress divergence of an isotropic moment tensor
    (Δσ_xx = Δσ_zz = M₀·s(t)·δ(z−z_s)), producing an isotropic radiation
    pattern identical to Devito's stress-injection monopole.

    For a line source uniform in x, ∂Δσ_xx/∂x = 0, so only uz is excited
    (through ∂Δσ_zz/∂z).  The result is a vertically propagating plane
    P-wave, matching the FK normal-incidence P-wave solution.

    Parameters
    ----------
    iz : int
        Z-index on the extended grid (source depth row).
    wavelet : ndarray, shape (nt,)
        Source time function amplitude at each time step.
    """
    iz: int
    wavelet: np.ndarray


@dataclass
class ExplosivePointSource:
    """Isotropic explosion point source (moment-tensor monopole).

    Equivalent to a single explosion at (ix_ext, iz).  Injects via the
    stress divergence of an isotropic moment tensor
    (Δσ_xx = Δσ_zz = M₀·s(t)·δ(x−x_s)δ(z−z_s)), producing an isotropic
    radiation pattern matching Devito's ``RickerPointSource`` tau injection.

    Parameters
    ----------
    ix_ext : int
        X-index on the extended grid (including PML padding).
    iz : int
        Z-index on the extended grid (source depth row).
    wavelet : ndarray, shape (nt,)
        Source time function amplitude at each time step.
    """
    ix_ext: int
    iz: int
    wavelet: np.ndarray


# ===================================================================
# Source injection
# ===================================================================

def inject_source(solver: FDDisplacementSolver, source, it: int) -> None:
    """Inject source into the solver's ``u_new`` fields.

    Call after ``compute_new`` / ``apply_boundary`` and before ``rotate``.

    Supports three source types:

    - ``PointBodyForceSource`` / ``LineBodyForceSource``: adds
      dt²·f(t)/ρ equally to ux_new and uz_new (body-force injection,
      dipole radiation pattern).
    - ``ExplosiveLineSource``: injects the stress divergence of an
      isotropic moment tensor Δσ_xx = Δσ_zz = M₀·s(t)·δ(z−z_s).
      For a line source uniform in x, ∂Δσ_xx/∂x = 0, so only uz is
      excited via ∂Δσ_zz/∂z (isotropic/monopole radiation).
    - ``ExplosivePointSource``: same moment-tensor injection at a single
      grid point; excites both ux (∂Δσ_xx/∂x) and uz (∂Δσ_zz/∂z).
    """
    if it >= len(source.wavelet):
        return
    amp = source.wavelet[it]
    if amp == 0.0:
        return
    dt2 = solver.dt ** 2

    if isinstance(source, ExplosiveLineSource):
        _inject_explosive_line(solver, source.iz, dt2 * amp)
    elif isinstance(source, ExplosivePointSource):
        _inject_explosive_point(solver, source.ix_ext, source.iz, dt2 * amp)
    elif isinstance(source, LineBodyForceSource):
        for ix_ext in range(solver.npml, solver.npml + solver.nx):
            scale = dt2 * amp / solver.rho[source.iz, ix_ext]
            solver.ux_new[source.iz, ix_ext] += scale
            solver.uz_new[source.iz, ix_ext] += scale
    else:
        scale = dt2 * amp / solver.rho[source.iz, source.ix_ext]
        solver.ux_new[source.iz, source.ix_ext] += scale
        solver.uz_new[source.iz, source.ix_ext] += scale


def _inject_explosive_line(
    solver: FDDisplacementSolver, iz_s: int, dt2_amp: float,
) -> None:
    """Moment-tensor injection for an isotropic explosion line source.

    Computes the FD z-derivative of the stress perturbation
    Δσ_zz = M₀·s(t)·δ(z − z_s) and adds the resulting acceleration
    to uz_new.  Uses the same 1st-derivative FD stencil as the solver.
    """
    idz = 1.0 / solver.dz
    norder = solver.norder
    ix_lo = solver.npml
    ix_hi = solver.npml + solver.nx

    for k in range(norder):
        ck = solver.coef[k, norder - 1]
        iz_plus = iz_s + k + 1
        iz_minus = iz_s - k - 1

        if iz_plus < solver.nz_tot:
            for ix in range(ix_lo, ix_hi):
                solver.uz_new[iz_plus, ix] -= (
                    dt2_amp * ck * idz / solver.rho[iz_plus, ix]
                )

        if iz_minus >= 0:
            for ix in range(ix_lo, ix_hi):
                solver.uz_new[iz_minus, ix] += (
                    dt2_amp * ck * idz / solver.rho[iz_minus, ix]
                )


def _inject_explosive_point(
    solver: FDDisplacementSolver, ix_s: int, iz_s: int, dt2_amp: float,
) -> None:
    """Moment-tensor injection for an isotropic explosion point source.

    Computes FD derivatives of Δσ_xx and Δσ_zz at (ix_s, iz_s) and adds
    the resulting accelerations to ux_new and uz_new respectively.
    """
    idx = 1.0 / solver.dx
    idz = 1.0 / solver.dz
    norder = solver.norder

    for k in range(norder):
        ck = solver.coef[k, norder - 1]

        ix_plus = ix_s + k + 1
        ix_minus = ix_s - k - 1
        if ix_plus < solver.nx_tot:
            solver.ux_new[iz_s, ix_plus] -= (
                dt2_amp * ck * idx / solver.rho[iz_s, ix_plus]
            )
        if ix_minus >= 0:
            solver.ux_new[iz_s, ix_minus] += (
                dt2_amp * ck * idx / solver.rho[iz_s, ix_minus]
            )

        iz_plus = iz_s + k + 1
        iz_minus = iz_s - k - 1
        if iz_plus < solver.nz_tot:
            solver.uz_new[iz_plus, ix_s] -= (
                dt2_amp * ck * idz / solver.rho[iz_plus, ix_s]
            )
        if iz_minus >= 0:
            solver.uz_new[iz_minus, ix_s] += (
                dt2_amp * ck * idz / solver.rho[iz_minus, ix_s]
            )


# ===================================================================
# Standalone time-marching runner
# ===================================================================

def run_standalone(
    solver: FDDisplacementSolver,
    source,
    nt: int,
    receivers_ix: np.ndarray | None = None,
    receivers_iz: np.ndarray | None = None,
) -> dict:
    """Time-march the FD solver with an internal source.

    This is a validation-only utility (not used by the hybrid engine).
    It calls compute_new() -> apply_boundary() -> inject_source() -> rotate()
    at each time step and records displacement at receiver locations.

    Parameters
    ----------
    solver : FDDisplacementSolver
        Initialized solver (material arrays, PML, etc. already set up).
    source : PointBodyForceSource, LineBodyForceSource, ExplosiveLineSource,
        or ExplosivePointSource
        Source to inject.
    nt : int
        Number of time steps.
    receivers_ix : ndarray of int, optional
        Extended-grid x-indices of receivers. If None, records at all
        regular-domain surface points.
    receivers_iz : ndarray of int, optional
        Extended-grid z-indices of receivers. If None, uses iz=0 (surface).

    Returns
    -------
    dict with keys:
        'ux' : ndarray, shape (nrcv, nt) — horizontal displacement
        'uz' : ndarray, shape (nrcv, nt) — vertical displacement
        't'  : ndarray, shape (nt,) — time axis
    """
    solver.reset()

    if receivers_ix is None:
        receivers_ix = np.arange(solver.npml, solver.npml + solver.nx)
    if receivers_iz is None:
        receivers_iz = np.zeros(len(receivers_ix), dtype=int)

    nrcv = len(receivers_ix)
    seis_ux = np.zeros((nrcv, nt), dtype=np.float64)
    seis_uz = np.zeros((nrcv, nt), dtype=np.float64)

    for it in range(nt):
        solver.compute_new()
        solver.apply_boundary()
        inject_source(solver, source, it)
        solver.rotate()

        for ir in range(nrcv):
            seis_ux[ir, it] = solver.ux_cur[receivers_iz[ir], receivers_ix[ir]]
            seis_uz[ir, it] = solver.uz_cur[receivers_iz[ir], receivers_ix[ir]]

    return {
        'ux': seis_ux,
        'uz': seis_uz,
        't': np.arange(nt) * solver.dt,
    }
