#!/usr/bin/env python3
"""
Shared model builder for AT variant cases.

The builder keeps the original AT_single_ice geometry/source setup and only
changes layering logic according to each variant specification.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# =============================================================================
# Shared constants (kept consistent with AT_single_ice defaults)
# =============================================================================

USE_SYNTHETIC_ICETHICKNESS = False

BEDMACHINE_NC_PATH = Path(
    "/Users/jygong/Stanford/Research_Data/0_IceModel/BedMachine_V4/"
    "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
)
BEDMACHINE_CODE_PATH = Path(
    "/Users/jygong/Stanford/Research_Data/0_IceModel/BedMachine_V4/scripts_bedmachine"
)

ICE_VP = 4000.0
ICE_VS = 2000.0
LOW_ICE_VP = 4000.0
LOW_ICE_VS = 1500.0
ROCK_VP = 4500.0
ROCK_VS = ROCK_VP / 1.75
CRUST_VP = 6300.0
CRUST_VS = 3600.0
SED_VP = 4600.0
SED_VS = 1800.0
WATER_VP = 1500.0
WATER_VS = 100.0

CENTER_LAT = -80.4178
CENTER_LON = 77.1111
PROFILE_AZ_DEG = 30.0
PROFILE_LENGTH_M = 15_000.0
DX = DZ = 50.0
NPML = 100
X0 = 0.0
XN = X0 + PROFILE_LENGTH_M
EARTH_R_M = 6_371_000.0

GAUSS_SIGMA_SAMPLES = 2.0
MIN_ZN_M = 12_000.0
ICE_BUFFER_M = 5_000.0
DT_S = 0.003
TMAX_S = 20.0

SOURCE_F0_HZ = 5.0
SOURCE_STRENGTH_STR = "1.d-3"
SOURCE_IS_P_WAVE = True  # P wave in Source.dat (Fortran: .true.)
TELESEISMIC_DELTA_DEG = 58.0
TELESEISMIC_SOURCE_DEPTH_KM = 10.0
TELESEISMIC_AZIMUTH_STATION_TO_EVENT_DEG = PROFILE_AZ_DEG

_IASP91_P_DT_DDELTA_DEG = (
    (30.0, 5.82),
    (40.0, 6.20),
    (50.0, 6.46),
    (60.0, 6.73),
    (70.0, 6.98),
    (90.0, 7.25),
)


@dataclass(frozen=True)
class LayerProps:
    """Elastic properties used in the current stage (rho handled by Fortran)."""

    vp: float
    vs: float


@dataclass(frozen=True)
class VariantSpec:
    """Definition of one AT variant."""

    case_name: str
    double_ice: bool
    basement: LayerProps
    include_sediment: bool = False
    sediment_half_width_m: float = 500.0
    sediment_max_thickness_m: float = 300.0
    include_water: bool = False
    water_half_width_m: float = 300.0
    water_max_thickness_m: float = 100.0
    water_surface_offset_m: float = 100.0


def profile_origin_latlon() -> tuple[float, float]:
    """Geographic position at FD origin (x = 0 m), z = 0 is ice-air surface."""
    phi_c = math.radians(CENTER_LAT)
    br = math.radians(PROFILE_AZ_DEG)
    s = -0.5 * PROFILE_LENGTH_M
    d_north = s * math.cos(br)
    d_east = s * math.sin(br)
    lat0 = CENTER_LAT + math.degrees(d_north / EARTH_R_M)
    lon0 = CENTER_LON + math.degrees(d_east / (EARTH_R_M * math.cos(phi_c)))
    return lat0, lon0


def destination_latlon(
    lat0_deg: float,
    lon0_deg: float,
    bearing_deg: float,
    distance_deg: float,
) -> tuple[float, float]:
    """Great-circle destination on a sphere."""
    delta = math.radians(distance_deg)
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat0_deg)
    lam1 = math.radians(lon0_deg)
    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    lon2 = math.degrees(lam2)
    lon2 = ((lon2 + 180.0) % 360.0) - 180.0
    return math.degrees(phi2), lon2


def great_circle_distance_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Central angle [deg] between two surface points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    c = min(1.0, max(-1.0, c))
    return math.degrees(math.acos(c))


def iasp91_p_rayparam_s_per_deg(delta_deg: float, depth_km: float) -> float:
    """Return iasp91 P horizontal slowness in s/deg."""
    try:
        from obspy.taup import TauPyModel  # type: ignore

        model = TauPyModel(model="iasp91")
        arrivals = model.get_travel_times(
            source_depth_in_km=depth_km,
            distance_in_degree=delta_deg,
            phase_list=["P"],
        )
        if not arrivals:
            raise RuntimeError("No P arrival from iasp91 for this distance/depth.")
        p_rad = float(arrivals[0].ray_param)
        return p_rad * (math.pi / 180.0)
    except Exception:
        xs = [a[0] for a in _IASP91_P_DT_DDELTA_DEG]
        ys = [a[1] for a in _IASP91_P_DT_DDELTA_DEG]
        return float(np.interp(delta_deg, xs, ys))


def write_source_dat(out_dir: Path) -> None:
    """Write Source.dat using shared teleseismic settings."""
    st_lat, st_lon = profile_origin_latlon()
    ev_lat, ev_lon = destination_latlon(
        st_lat,
        st_lon,
        TELESEISMIC_AZIMUTH_STATION_TO_EVENT_DEG,
        TELESEISMIC_DELTA_DEG,
    )
    delta_chk = great_circle_distance_deg(st_lat, st_lon, ev_lat, ev_lon)
    if abs(delta_chk - TELESEISMIC_DELTA_DEG) > 0.05:
        raise RuntimeError(f"Great-circle distance mismatch: {delta_chk} vs {TELESEISMIC_DELTA_DEG}")

    rayp = iasp91_p_rayparam_s_per_deg(TELESEISMIC_DELTA_DEG, TELESEISMIC_SOURCE_DEPTH_KM)
    pflag = ".true." if SOURCE_IS_P_WAVE else ".false."
    text = (
        "f0(Hz)                           streng of source(Pa)        source type(P: .true.; S: .false.)\n"
        f"   {SOURCE_F0_HZ:.1f}                           {SOURCE_STRENGTH_STR}                       {pflag}\n"
        "ray parameter(s/deg)             latitude of source(deg)     longtitude of source(deg)\n"
        f"   {rayp:.6f}                           {ev_lat:.6f}                       {ev_lon:.6f}\n"
    )
    (out_dir / "Source.dat").write_text(text, encoding="utf-8")


def latlon_along_profile(x_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map FD x positions (m) to geographic lat/lon."""
    lat0, lon0 = profile_origin_latlon()
    phi0 = math.radians(lat0)
    br = math.radians(PROFILE_AZ_DEG)
    d_north = x_m * math.cos(br)
    d_east = x_m * math.sin(br)
    lat = lat0 + np.degrees(d_north / EARTH_R_M)
    lon = lon0 + np.degrees(d_east / (EARTH_R_M * math.cos(phi0)))
    return lat, lon


def write_su_depth(path: Path, data: np.ndarray, dz: float) -> None:
    """Write stream SU in depth domain."""
    if data.ndim != 2:
        raise ValueError("data must be 2-D (nz, nx)")
    nz, nx = data.shape
    head = np.zeros(120, dtype=np.uint16)
    head[57] = int(nz)
    head[58] = int(round(dz * 1000.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for ix in range(nx):
            f.write(head.tobytes())
            col = np.ascontiguousarray(data[:, ix], dtype=np.float32)
            f.write(col.tobytes())


def ice_thickness_bedmachine(
    lats: np.ndarray,
    lons: np.ndarray,
    nc_path: Path,
    code_path: Path,
) -> np.ndarray:
    """Sample BedMachine ice thickness along profile."""
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))
    from interp_bedmachine_antarctica import interp_bedmachine_antarctica  # type: ignore
    from ll2xy import ll2xy  # type: ignore

    lats = np.asarray(lats, dtype=float).ravel()
    lons = np.asarray(lons, dtype=float).ravel()
    xy = np.array([ll2xy(float(lat), float(lon), -1) for lat, lon in zip(lats, lons)], dtype=float)
    surf = np.asarray(
        interp_bedmachine_antarctica(
            xy[:, 0], xy[:, 1], "surface", return_grid=False, bedmachine_nc_path=str(nc_path)
        )
    ).ravel()
    bed = np.asarray(
        interp_bedmachine_antarctica(
            xy[:, 0], xy[:, 1], "bed", return_grid=False, bedmachine_nc_path=str(nc_path)
        )
    ).ravel()
    return surf - bed


def ice_thickness_synthetic(x_m: np.ndarray) -> np.ndarray:
    """Smooth fake thickness for offline tests."""
    return 800.0 + 200.0 * np.sin(2.0 * math.pi * x_m / PROFILE_LENGTH_M)


def separable_gaussian_smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    """Lightweight separable Gaussian (edge padding); no SciPy dependency."""
    if sigma <= 0:
        return data.copy()
    rad = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    k /= np.sum(k)

    def conv1d_axis0(a: np.ndarray) -> np.ndarray:
        pad = np.pad(a, ((rad, rad), (0, 0)), mode="edge")
        out = np.empty_like(a, dtype=np.float64)
        for j in range(a.shape[1]):
            out[:, j] = np.convolve(pad[:, j], k, mode="valid")
        return out

    def conv1d_axis1(a: np.ndarray) -> np.ndarray:
        pad = np.pad(a, ((0, 0), (rad, rad)), mode="edge")
        out = np.empty_like(a, dtype=np.float64)
        for i in range(a.shape[0]):
            out[i, :] = np.convolve(pad[i, :], k, mode="valid")
        return out

    return conv1d_axis1(conv1d_axis0(np.asarray(data, dtype=np.float64)))


def write_fd_model(out_dir: Path, zn_m: float) -> None:
    """Write FD_model.dat."""
    lat0, lon0 = profile_origin_latlon()
    lines = [
        "order of FD(2m>=2)",
        "  6",
        "X0(m)        Xn(m)        Z0(m)     Zn(m)       Dx(m)      Dz(m)       Dt(s)",
        f"  {X0:g}          {XN:g}        0         {zn_m:g}       {DX:g}         {DZ:g}          {DT_S:g}",
        "tmax(s)      tstep        nPML(number of grid)  is_PML_top(.true./.false.: top boundary is PMLs?)",
        f"  {TMAX_S:g}       {DT_S:g}        {NPML}                    .false.",
        "latitude of (X0,Z0)       longitude of (X0,Z0)  azimuthe of profile",
        f"  {lat0:.14f}      {lon0:.14f}    {PROFILE_AZ_DEG:.1f}",
        "output snapshot?(.true./.false.)  nstep(output snapshot every nstep steps)",
        "  .true.                          100",
        "",
    ]
    (out_dir / "FD_model.dat").write_text("\n".join(lines), encoding="utf-8")


def write_receiver_dat(out_dir: Path) -> None:
    """Write default 101 surface receivers with 100 m spacing."""
    lines = ["number of receivers:", "     101", "rx(m)   rz(m)"]
    for rx in range(0, 10_001, 100):
        lines.append(f"{rx}\t0")
    lines.append("")
    (out_dir / "Receiver.dat").write_text("\n".join(lines), encoding="utf-8")


def quadratic_taper(distance_from_center_m: np.ndarray, half_width_m: float) -> np.ndarray:
    """
    Smooth quadratic taper: 1 at center and 0 at +/- half_width.
    Returns zeros outside the half-width.
    """
    if half_width_m <= 0:
        raise ValueError("half_width_m must be positive.")
    ratio = np.abs(distance_from_center_m) / half_width_m
    taper = 1.0 - ratio**2
    return np.clip(taper, 0.0, 1.0)


def build_h_profile(x_phys: np.ndarray) -> np.ndarray:
    """Build raw ice thickness profile along physical x."""
    if USE_SYNTHETIC_ICETHICKNESS:
        h_phys = ice_thickness_synthetic(x_phys)
    else:
        if not BEDMACHINE_NC_PATH.is_file():
            raise FileNotFoundError(
                "BedMachine NetCDF not found. Set BEDMACHINE_NC_PATH in at_variant_builder.py.\n"
                f"  Tried: {BEDMACHINE_NC_PATH}"
            )
        if not BEDMACHINE_CODE_PATH.is_dir():
            raise FileNotFoundError(
                "BedMachine scripts directory not found. Set BEDMACHINE_CODE_PATH in at_variant_builder.py.\n"
                f"  Tried: {BEDMACHINE_CODE_PATH}"
            )
        lats, lons = latlon_along_profile(x_phys)
        h_phys = ice_thickness_bedmachine(lats, lons, BEDMACHINE_NC_PATH, BEDMACHINE_CODE_PATH)
    return np.clip(h_phys, 10.0, None)


def assign_layers_for_column(
    z_edges: np.ndarray,
    ice_h: float,
    interbed_h: float,
    spec: VariantSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign Vp/Vs for one vertical column."""
    vp_col = np.full_like(z_edges, spec.basement.vp, dtype=float)
    vs_col = np.full_like(z_edges, spec.basement.vs, dtype=float)

    ice_top = 0.0
    ice_bottom = max(ice_h, 0.0)
    if spec.double_ice:
        upper_bottom = ice_top + (2.0 / 3.0) * ice_bottom
        upper_mask = (z_edges >= ice_top) & (z_edges < upper_bottom)
        lower_mask = (z_edges >= upper_bottom) & (z_edges < ice_bottom)
        vp_col[upper_mask] = ICE_VP
        vs_col[upper_mask] = ICE_VS
        vp_col[lower_mask] = LOW_ICE_VP
        vs_col[lower_mask] = LOW_ICE_VS
    else:
        ice_mask = (z_edges >= ice_top) & (z_edges < ice_bottom)
        vp_col[ice_mask] = ICE_VP
        vs_col[ice_mask] = ICE_VS

    if interbed_h > 0.0:
        interbed_top = ice_bottom
        interbed_bottom = interbed_top + interbed_h
        interbed_mask = (z_edges >= interbed_top) & (z_edges < interbed_bottom)
        if spec.include_sediment:
            vp_col[interbed_mask] = SED_VP
            vs_col[interbed_mask] = SED_VS
        elif spec.include_water:
            vp_col[interbed_mask] = WATER_VP
            vs_col[interbed_mask] = WATER_VS

    return vp_col, vs_col


def layer_boundaries_for_x(ice_h: float, interbed_h: float, spec: VariantSpec) -> list[tuple[LayerProps, float]]:
    """Return layer sequence for FK file generation."""
    layers: list[tuple[LayerProps, float]] = []
    if spec.double_ice:
        layers.append((LayerProps(ICE_VP, ICE_VS), 0.0))
        layers.append((LayerProps(LOW_ICE_VP, LOW_ICE_VS), (2.0 / 3.0) * ice_h))
    else:
        layers.append((LayerProps(ICE_VP, ICE_VS), 0.0))

    if spec.include_sediment and interbed_h > 0.0:
        layers.append((LayerProps(SED_VP, SED_VS), ice_h))
        layers.append((spec.basement, ice_h + interbed_h))
    elif spec.include_water and interbed_h > 0.0:
        layers.append((LayerProps(WATER_VP, WATER_VS), ice_h))
        layers.append((spec.basement, ice_h + interbed_h))
    else:
        layers.append((spec.basement, ice_h))
    return layers


def write_fk_file(path: Path, layers: list[tuple[LayerProps, float]]) -> None:
    """Write one FK model file from layer definitions."""
    lines = [
        "nlayer",
        f"  {len(layers)}",
        "Vp(m/s)   Vs(m/s)   z+(m) <----- top boundary of the layers",
    ]
    for props, z_top in layers:
        lines.append(f"{props.vp:.1f}    {props.vs:.6f}    {z_top:.3f}d0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_fk_files_for_case(
    out_dir: Path,
    h_left_m: float,
    h_right_m: float,
    interbed_left_m: float,
    interbed_right_m: float,
    spec: VariantSpec,
) -> None:
    """Build left/right FK files with case-specific layering."""
    left_layers = layer_boundaries_for_x(h_left_m, interbed_left_m, spec)
    right_layers = layer_boundaries_for_x(h_right_m, interbed_right_m, spec)
    write_fk_file(out_dir / "FK_model_left.dat", left_layers)
    write_fk_file(out_dir / "FK_model_right.dat", right_layers)


def build_variant_case(output_dir: Path, spec: VariantSpec) -> None:
    """Main entry point for one AT variant case."""
    out_dir = output_dir.resolve()

    nx_phys = int(round((XN - X0) / DX)) + 1
    x_phys = np.linspace(X0, XN, nx_phys, dtype=float)
    center_x = 0.5 * (X0 + XN)

    h_phys_raw = build_h_profile(x_phys)
    h_center = float(np.interp(center_x, x_phys, h_phys_raw))
    h_phys_effective = h_phys_raw.copy()

    if spec.include_water:
        lake_ice_h = h_center - spec.water_surface_offset_m
        if lake_ice_h <= 1.0:
            raise ValueError(
                "Computed lake-zone ice thickness is non-positive. "
                f"center_h={h_center:.2f}, offset={spec.water_surface_offset_m:.2f}"
            )
        in_lake = np.abs(x_phys - center_x) <= spec.water_half_width_m
        h_phys_effective[in_lake] = lake_ice_h

    max_h = float(np.max(h_phys_effective))
    max_interbed = max(spec.sediment_max_thickness_m if spec.include_sediment else 0.0, spec.water_max_thickness_m if spec.include_water else 0.0)
    zn_m = max(MIN_ZN_M, math.ceil((max_h + max_interbed + ICE_BUFFER_M) / DZ) * DZ)

    nz_phys = int(round((zn_m - 0.0) / DZ)) + 1
    nz_su = nz_phys + NPML
    nx_su = nx_phys + 2 * NPML

    ix_fortran = np.arange(1 - NPML, nx_phys + NPML + 1, dtype=int)
    x_m = X0 + (ix_fortran.astype(float) - 1.0) * DX
    x_m = np.clip(x_m, X0, XN)

    h_cols_effective = np.interp(x_m, x_phys, h_phys_effective)
    h_cols_raw = np.interp(x_m, x_phys, h_phys_raw)

    interbed_cols = np.zeros_like(h_cols_effective)
    if spec.include_sediment:
        h_center_ref = max(float(np.interp(center_x, x_phys, h_phys_effective)), 1.0)
        taper = quadratic_taper(x_m - center_x, spec.sediment_half_width_m)
        interbed_cols = (spec.sediment_max_thickness_m / h_center_ref) * h_cols_effective * taper
    elif spec.include_water:
        h_center_ref = max(float(np.interp(center_x, x_phys, h_phys_effective)), 1.0)
        taper = quadratic_taper(x_m - center_x, spec.water_half_width_m)
        interbed_cols = (spec.water_max_thickness_m / h_center_ref) * h_cols_effective * taper

    z_edges = np.arange(nz_su, dtype=float) * DZ
    vp = np.zeros((nz_su, nx_su), dtype=np.float64)
    vs = np.zeros_like(vp)
    for j in range(nx_su):
        vp_col, vs_col = assign_layers_for_column(
            z_edges=z_edges,
            ice_h=float(h_cols_effective[j]),
            interbed_h=float(interbed_cols[j]),
            spec=spec,
        )
        vp[:, j] = vp_col
        vs[:, j] = vs_col

    vp_s = separable_gaussian_smooth(vp, GAUSS_SIGMA_SAMPLES)
    vs_s = separable_gaussian_smooth(vs, GAUSS_SIGMA_SAMPLES)

    write_fd_model(out_dir, zn_m)
    write_receiver_dat(out_dir)
    write_su_depth(out_dir / "vp_model_smooth_2x2.su", vp_s.astype(np.float32), DZ)
    write_su_depth(out_dir / "vs_model_smooth_2x2.su", vs_s.astype(np.float32), DZ)

    h_left = float(np.interp(0.0, x_phys, h_phys_effective))
    h_right = float(np.interp(XN, x_phys, h_phys_effective))
    interbed_left = float(np.interp(0.0, x_m, interbed_cols))
    interbed_right = float(np.interp(XN, x_m, interbed_cols))
    build_fk_files_for_case(out_dir, h_left, h_right, interbed_left, interbed_right, spec)
    write_source_dat(out_dir)

    print(f"{spec.case_name} model build OK")
    print(f"  max effective ice thickness (m): {max_h:.1f}")
    print(f"  max raw ice thickness (m): {float(np.max(h_phys_raw)):.1f}")
    print(f"  max interbed thickness (m): {float(np.max(interbed_cols)):.1f}")
    print(f"  Zn (m): {zn_m:g}  -> nz_phys={nz_phys}, nz_su={nz_su}")
    print(f"  nx_phys={nx_phys}, nx_su={nx_su}")
    print(f"  H(0)={h_left:.2f} m, H(Xn)={h_right:.2f} m")
    if spec.include_sediment or spec.include_water:
        print(
            f"  interbed(0)={interbed_left:.2f} m, interbed(Xn)={interbed_right:.2f} m, "
            f"interbed(center)={float(np.interp(center_x, x_m, interbed_cols)):.2f} m"
        )
    if spec.include_water:
        print(
            "  lake-zone ice thickness fixed in +/- "
            f"{spec.water_half_width_m:.0f} m around center."
        )
