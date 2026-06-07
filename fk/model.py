"""1D layered velocity model for the FK method.

A model is a stack of homogeneous, isotropic, elastic layers over a bottom
half-space. Each entry stores P velocity, S velocity, density and the depth of
its top boundary. The last entry is the half-space (infinite thickness).

This matches the FK model files of the original FDFK2D package, where each row
is ``Vp  Vs  z_top`` and ``nlayer`` counts all rows including the half-space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def density_from_vp_fdfk(vp: np.ndarray) -> np.ndarray:
    """Density from P velocity using the FDFK2D linear relation.

    Not Gardner's (1974) power-law ``rho ~ Vp^b``; this matches Input.f90:
    ``rho = (Vp + 980) / 2.760`` with ``Vp`` in m/s, returning density in kg/m^3.

    Parameters
    ----------
    vp : array_like
        P-wave velocity in m/s.

    Returns
    -------
    numpy.ndarray
        Density in kg/m^3.
    """
    vp = np.asarray(vp, dtype=float)
    return (vp + 980.0) / 2.760


@dataclass
class FKLayerModel:
    """1D layered isotropic elastic model (last entry = half-space).

    Parameters
    ----------
    vp, vs : numpy.ndarray
        P- and S-wave velocities (m/s), one value per entry.
    rho : numpy.ndarray
        Density (kg/m^3), one value per entry.
    z_top : numpy.ndarray
        Depth of the top boundary of each entry (m), increasing. ``z_top[0]``
        is the free surface (usually 0). The last entry is the half-space.
    """

    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    z_top: np.ndarray
    mu: np.ndarray = field(init=False)
    lam: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.vp = np.atleast_1d(np.asarray(self.vp, dtype=float))
        self.vs = np.atleast_1d(np.asarray(self.vs, dtype=float))
        self.rho = np.atleast_1d(np.asarray(self.rho, dtype=float))
        self.z_top = np.atleast_1d(np.asarray(self.z_top, dtype=float))
        self._validate()
        # Lame parameters; mu = rho * vs^2, lambda = rho * (vp^2 - 2 vs^2).
        self.mu = self.rho * self.vs ** 2
        self.lam = self.rho * (self.vp ** 2 - 2.0 * self.vs ** 2)

    def _validate(self) -> None:
        n = self.vp.size
        if not (self.vs.size == n == self.rho.size == self.z_top.size):
            raise ValueError("vp, vs, rho, z_top must have the same length.")
        if n < 1:
            raise ValueError("Model must have at least one entry (half-space).")
        if np.any(self.vp <= 0):
            raise ValueError("All Vp values must be positive.")
        if np.any(self.vs <= 0):
            raise ValueError("All Vs values must be positive (FK P-SV needs Vs>0).")
        if np.any(self.vp < self.vs * np.sqrt(2.0)):
            raise ValueError("Require Vp >= sqrt(2)*Vs (non-negative Lame lambda).")
        if np.any(np.diff(self.z_top) <= 0) and n > 1:
            raise ValueError("z_top must be strictly increasing with depth.")

    @property
    def n_entries(self) -> int:
        """Total number of entries including the half-space."""
        return self.vp.size

    @property
    def n_finite_layers(self) -> int:
        """Number of finite-thickness layers (entries above the half-space)."""
        return self.vp.size - 1

    @property
    def thickness(self) -> np.ndarray:
        """Thickness of each finite layer (m); half-space gets ``inf``."""
        if self.n_entries == 1:
            return np.array([np.inf])
        h = np.diff(self.z_top)
        return np.append(h, np.inf)

    @property
    def halfspace_index(self) -> int:
        """Index of the half-space entry (the last one)."""
        return self.n_entries - 1

    @classmethod
    def from_layers(
        cls,
        vp,
        vs,
        z_top,
        rho=None,
    ) -> "FKLayerModel":
        """Build a model from velocities and top depths.

        If ``rho`` is not given, it is computed from :func:`density_from_vp_fdfk`.
        """
        vp = np.atleast_1d(np.asarray(vp, dtype=float))
        vs = np.atleast_1d(np.asarray(vs, dtype=float))
        z_top = np.atleast_1d(np.asarray(z_top, dtype=float))
        if rho is None:
            rho = density_from_vp_fdfk(vp)
        return cls(vp=vp, vs=vs, rho=rho, z_top=z_top)

    def summary(self) -> str:
        """Return a human-readable table of the model."""
        lines = ["  idx     Vp       Vs      rho     z_top   thick"]
        for i in range(self.n_entries):
            th = self.thickness[i]
            th_s = "inf" if not np.isfinite(th) else f"{th:8.1f}"
            tag = " (half-space)" if i == self.halfspace_index else ""
            lines.append(
                f"  {i:3d} {self.vp[i]:8.1f} {self.vs[i]:8.1f} "
                f"{self.rho[i]:8.1f} {self.z_top[i]:8.1f} {th_s}{tag}"
            )
        return "\n".join(lines)
