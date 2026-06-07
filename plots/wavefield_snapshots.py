"""Utilities for 2D wavefield snapshot figures and interface annotation."""

from __future__ import annotations

from typing import Sequence

import matplotlib.axes
import numpy as np


InterfaceProfile = tuple[np.ndarray, np.ndarray]

# Python hybrid snapshots include iz=0, a free-surface ghost row (extrapolated by
# _apply_free_surface, not FD-updated). Skip it for display; Fortran snapshots
# start at the physical surface (iz=1 in Fortran == z=0).
DEFAULT_HYBRID_IZ_PLOT_START = 1


def snapshot_for_plot(data: np.ndarray, iz_start: int = DEFAULT_HYBRID_IZ_PLOT_START) -> np.ndarray:
    """Drop leading depth rows before plotting (typically the surface ghost row)."""
    if iz_start <= 0:
        return data
    return data[iz_start:, :]


def extent_km_for_plot(
    x0: float,
    xn: float,
    z0: float,
    zn: float,
    dz: float,
    iz_start: int = DEFAULT_HYBRID_IZ_PLOT_START,
) -> list[float]:
    """Depth extent (km) matching :func:`snapshot_for_plot` row selection."""
    z_top_km = (z0 + iz_start * dz) / 1000.0
    return [x0 / 1000.0, xn / 1000.0, zn / 1000.0, z_top_km]


def model_interface_profiles(
    vs_model: np.ndarray,
    x0: float,
    dx: float,
    z0: float,
    dz: float,
    *,
    threshold_frac: float = 0.25,
    max_depth_km: float | None = None,
) -> list[InterfaceProfile]:
    """Extract laterally varying interface curves from a 2D Vs model.

    Interface locations are identified from sharp jumps in the laterally
    averaged Vs profile (same threshold rule as the previous horizontal
    ``axhline`` helper).  For each detected interface, the depth at every
    x column is found by matching the column's qualifying jump whose depth
    index is closest to the reference index from the mean profile.

    Parameters
    ----------
    vs_model
        2D S-wave velocity array with shape ``(nz, nx)``.
    x0, dx
        Grid origin and spacing along x (m).
    z0, dz
        Grid origin and spacing along z (m).
    threshold_frac
        Keep jumps whose magnitude is at least this fraction of the largest
        jump on the laterally averaged profile.
    max_depth_km
        If given, omit interfaces whose mean depth exceeds this value.

    Returns
    -------
    list of (x_km, depth_km)
        One profile per detected interface; arrays have length ``nx``.
    """
    vs_model = np.asarray(vs_model, dtype=float)
    if vs_model.ndim != 2:
        raise ValueError(f"vs_model must be 2-D, got shape {vs_model.shape}")

    nz, nx = vs_model.shape
    x_km = (x0 + np.arange(nx) * dx) / 1000.0

    vs_mean = np.nanmean(vs_model, axis=1)
    jumps_mean = np.abs(np.diff(vs_mean))
    if jumps_mean.size == 0 or np.nanmax(jumps_mean) <= 0:
        return []

    threshold = threshold_frac * float(np.nanmax(jumps_mean))
    iz_ref = np.where(jumps_mean >= threshold)[0] + 1
    if iz_ref.size == 0:
        return []

    profiles: list[InterfaceProfile] = []
    for iz_r in iz_ref:
        mean_depth_km = (z0 + iz_r * dz) / 1000.0
        if max_depth_km is not None and mean_depth_km > max_depth_km:
            continue

        depth_km = np.empty(nx, dtype=float)
        fallback_depth_km = mean_depth_km
        for ix in range(nx):
            col = vs_model[:, ix]
            jumps_col = np.abs(np.diff(col))
            candidates = np.where(jumps_col >= threshold)[0] + 1
            if candidates.size == 0:
                depth_km[ix] = fallback_depth_km
            else:
                best_iz = int(candidates[np.argmin(np.abs(candidates - iz_r))])
                depth_km[ix] = (z0 + best_iz * dz) / 1000.0
        profiles.append((x_km, depth_km))

    return profiles


def model_interface_depths(
    vs_model: np.ndarray,
    z0: float,
    dz: float,
    *,
    threshold_frac: float = 0.25,
    max_depth_km: float | None = None,
) -> list[float]:
    """Mean-profile interface depths in km (legacy horizontal helper)."""
    vs_model = np.asarray(vs_model, dtype=float)
    vs_mean = np.nanmean(vs_model, axis=1)
    jumps = np.abs(np.diff(vs_mean))
    if jumps.size == 0 or np.nanmax(jumps) <= 0:
        return []

    threshold = threshold_frac * float(np.nanmax(jumps))
    depths: list[float] = []
    for iz in np.where(jumps >= threshold)[0] + 1:
        depth_km = (z0 + iz * dz) / 1000.0
        if max_depth_km is not None and depth_km > max_depth_km:
            continue
        depths.append(depth_km)
    return depths


def annotate_interfaces(
    ax: matplotlib.axes.Axes,
    profiles: Sequence[InterfaceProfile],
    *,
    color: str = "0.15",
    lw: float = 0.55,
    alpha: float = 0.45,
    ls: str = "--",
) -> None:
    """Overlay dashed interface curves on a depth-section axis."""
    for x_km, depth_km in profiles:
        ax.plot(x_km, depth_km, color=color, lw=lw, alpha=alpha, ls=ls)
