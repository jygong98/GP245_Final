"""Geometric and wavenumber helpers for the FK method.

Includes:
- ray-parameter unit conversion (s/deg <-> s/m),
- incidence-angle and plane-wave initialization-depth helpers,
- vertical wavenumbers (kp_z, ks_z) and the auxiliary gamma term,
- a great-circle back-azimuth routine (ported from FDFK2D calc_backazimuth.f90)
  used by the file loader to reproduce the original geometry.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6371.0e3
EARTH_FLATTENING = 1.0 / 298.257


def rayp_deg_to_si(p_deg: float) -> float:
    """Convert ray parameter from s/deg to s/m (horizontal slowness).

    Mirrors Input.f90: ``rayp = rayp * 180 / (pi * 6371e3)``.
    """
    return p_deg * 180.0 / (np.pi * EARTH_RADIUS_M)


def rayp_si_to_deg(p_si: float) -> float:
    """Convert ray parameter from s/m to s/deg."""
    return p_si * np.pi * EARTH_RADIUS_M / 180.0


def incidence_angle(p_si: float, velocity: float) -> float:
    """Incidence angle (rad) of a plane wave with slowness ``p_si``.

    ``sin(theta) = p * v``. Raises if the wave is evanescent (p*v >= 1).
    """
    s = p_si * velocity
    if s >= 1.0:
        raise ValueError(
            f"Evanescent reference wave: p*v = {s:.4f} >= 1 "
            "(ray parameter too large for this velocity)."
        )
    return np.arcsin(s)


def init_depth(
    p_si: float,
    ref_velocity: float,
    profile_length: float,
    model_max_depth: float,
) -> float:
    """Plane-wave initialization depth ``z_o`` below the model.

    Follows the paper: ``z_o = z_n + (L/2) * tan(theta)`` where ``z_n`` is the
    maximum model depth, ``L`` the profile length and ``theta`` the incidence
    angle in the half-space.
    """
    theta = incidence_angle(p_si, ref_velocity)
    return model_max_depth + 0.5 * profile_length * np.tan(theta)


def vertical_wavenumbers(omega, p_si, vp, vs):
    """Vertical P/S wavenumbers and the gamma term for one layer.

    Parameters
    ----------
    omega : numpy.ndarray
        Angular frequencies (rad/s), shape (nfreq,).
    p_si : float
        Horizontal slowness (s/m); ``kh = omega * p``.
    vp, vs : float
        Layer P and S velocities (m/s).

    Returns
    -------
    kh, kp_z, ks_z, gamma : numpy.ndarray
        Complex arrays of shape (nfreq,). The complex square root uses the
        principal branch so evanescent waves decay with depth (Im >= 0).
    """
    omega = np.asarray(omega, dtype=float)
    kh = omega * p_si
    # kp_z^2 = omega^2/vp^2 - kh^2 ; use complex sqrt for evanescent safety.
    kp_z = np.sqrt((omega / vp) ** 2 - kh ** 2 + 0j)
    ks_z = np.sqrt((omega / vs) ** 2 - kh ** 2 + 0j)
    gamma = (omega / vs) ** 2 - 2.0 * kh ** 2
    return kh, kp_z, ks_z, gamma


def backazimuth(evla: float, evlo: float, stla: float, stlo: float) -> float:
    """Great-circle back-azimuth (deg) from station to event.

    Ported from the ``backaz`` subroutine in calc_backazimuth.f90 (Bullen,
    via T. Owens). ``baz`` is the azimuth from the event to the station,
    measured clockwise from north, returned in degrees in [0, 360).

    Parameters
    ----------
    evla, evlo : float
        Event latitude/longitude (deg).
    stla, stlo : float
        Station latitude/longitude (deg).
    """
    d2r = np.pi / 180.0
    r2d = 180.0 / np.pi
    coeff = (1.0 - EARTH_FLATTENING) ** 2

    scola = np.pi / 2.0 - np.arctan(coeff * np.tan(stla * d2r))
    ecola = np.pi / 2.0 - np.arctan(coeff * np.tan(evla * d2r))
    slon = stlo * d2r
    elon = evlo * d2r

    a = np.sin(scola) * np.cos(slon)
    b = np.sin(scola) * np.sin(slon)
    c = np.cos(scola)
    d = np.sin(slon)
    e = -np.cos(slon)
    g = -c * e
    h = c * d
    k = -np.sin(scola)

    aa = np.sin(ecola) * np.cos(elon)
    bb = np.sin(ecola) * np.sin(elon)
    cc = np.cos(ecola)

    rhs1 = (aa - d) ** 2 + (bb - e) ** 2 + cc ** 2 - 2.0
    rhs2 = (aa - g) ** 2 + (bb - h) ** 2 + (cc - k) ** 2 - 2.0
    dbaz = np.arctan2(rhs1, rhs2)
    if dbaz < 0.0:
        dbaz += 2.0 * np.pi
    baz = dbaz * r2d
    if abs(baz - 360.0) < 1.0e-5:
        baz = 0.0
    return baz
