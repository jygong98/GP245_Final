"""FDFK2D-Python: Hybrid FD-FK module.

Couples the FK propagator-matrix method with a 2D finite-difference solver
using TF/SF wavefield injection (Liu et al. 2025 / Tong et al. 2014).

Main entry points
-----------------
- :func:`hybrid.engine.compute_hybrid_seismograms` — single-run API
- :class:`hybrid.engine.HybridEngine` — batch API (caches FK between runs)
- :func:`hybrid.background.compute_fk_greens` — pre-compute FK Green's functions
"""

from .config import HybridConfig
from .engine import compute_hybrid_seismograms, HybridResult, HybridEngine
from .background import (
    compute_fk_greens,
    boundary_wavefield_from_greens,
    FKGreensCache,
    BoundaryWavefield,
    compute_regular_wavefield,
)

__all__ = [
    "HybridConfig",
    "compute_hybrid_seismograms",
    "HybridResult",
    "HybridEngine",
    "compute_fk_greens",
    "boundary_wavefield_from_greens",
    "FKGreensCache",
    "BoundaryWavefield",
    "compute_regular_wavefield",
]
