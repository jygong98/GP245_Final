"""FDFK2D-Python: FK (frequency-wavenumber / plane-wave) module.

Standalone Python implementation of the FK method for teleseismic plane-wave
modeling in 1D layered media, following Liu et al. (2025, SRL) Appendix A.

The package computes free-surface (or arbitrary-depth) displacement seismograms
(u_x, u_z) for an incident P or S plane wave traveling through a layered model.

Main entry points
-----------------
- :class:`fk.model.FKLayerModel` : 1D layered velocity model.
- :class:`fk.source.RickerSource` : source time function / incidence config.
- :func:`fk.engine.compute_fk_seismograms` : top-level FK seismogram computation.

Propagator methods (``fk.propagators``)
---------------------------------------
Pluggable frequency-domain solvers; select via ``method`` in
:func:`compute_fk_seismograms`:

- Kennett reflectivity (``kennett``, ``kennett_numba``; default).
- Thomson-Haskell propagator matrix (``haskell``).
"""

from .model import FKLayerModel
from .source import RickerSource
from .engine import FKResult, compute_fk_seismograms
from .propagators import (
    surface_state_kennett,
    state_at_depth_kennett,
    surface_state_numba,
    states_at_depths_numba,
    surface_state_kennett_numba,
    states_at_depths_kennett_numba,
)

__all__ = [
    "FKLayerModel",
    "RickerSource",
    "FKResult",
    "compute_fk_seismograms",
    # Thomson-Haskell (Numba; None if numba unavailable)
    "surface_state_numba",
    "states_at_depths_numba",
    # Kennett reflectivity
    "surface_state_kennett",
    "state_at_depth_kennett",
    "surface_state_kennett_numba",
    "states_at_depths_kennett_numba",
]
