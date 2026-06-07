"""Receiver-function computation from FK or FDFK seismograms."""

from .decon import compute_prf, compute_srf

__all__ = ["compute_prf", "compute_srf"]
