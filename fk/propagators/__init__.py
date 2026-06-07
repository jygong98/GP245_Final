"""Pluggable frequency-domain propagators for the FK plane-wave method.

Each module implements the same surface/depth state API for a 1D layered model
at fixed horizontal slowness. Select via ``method`` in
:func:`fk.engine.compute_fk_seismograms`.

- :mod:`fk.propagators.haskell` — Thomson-Haskell propagator matrix (NumPy).
- :mod:`fk.propagators.haskell_numba` — Haskell kernels (Numba).
- :mod:`fk.propagators.kennett` — Kennett reflectivity (NumPy).
- :mod:`fk.propagators.kennett_numba` — Kennett reflectivity (Numba; default).
"""

from .haskell import (
    build_propagator,
    eigen_matrix_R,
    incident_amplitude,
    layer_propagator_P,
    phase_matrix_H,
    solve_halfspace_coeffs,
    state_at_depth,
    surface_state,
)
from .kennett import state_at_depth_kennett, surface_state_kennett

__all__ = [
    # Haskell (NumPy)
    "build_propagator",
    "eigen_matrix_R",
    "incident_amplitude",
    "layer_propagator_P",
    "phase_matrix_H",
    "solve_halfspace_coeffs",
    "state_at_depth",
    "surface_state",
    # Kennett (NumPy)
    "state_at_depth_kennett",
    "surface_state_kennett",
    # Numba (optional)
    "surface_state_numba",
    "states_at_depths_numba",
    "surface_state_kennett_numba",
    "states_at_depths_kennett_numba",
]

try:
    from .haskell_numba import surface_state_numba, states_at_depths_numba
    from .kennett_numba import (
        surface_state_kennett_numba,
        states_at_depths_kennett_numba,
    )
except ImportError:  # numba not installed
    surface_state_numba = None
    states_at_depths_numba = None
    surface_state_kennett_numba = None
    states_at_depths_kennett_numba = None
