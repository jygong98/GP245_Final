"""YAML configuration loader for FK and hybrid FD-FK simulations.

Human-editable alternative to the Fortran ``*.dat`` parameter decks. Example
configs live in ``examples/config/``.

Quick start
-----------
>>> from fk.io_yaml import load_yaml
>>> sim = load_yaml("examples/config/test1_fk.yaml")
>>> sim.config          # HybridConfig (grid, time, PML)
>>> sim.model_left      # FKLayerModel
>>> sim.source          # RickerSource
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from hybrid.config import HybridConfig

from .geometry import backazimuth, init_depth, rayp_deg_to_si
from .io_fortran import read_fk_model
from .model import FKLayerModel
from .source import RickerSource


def _import_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML config loading. "
            "Install with: pip install pyyaml"
        ) from exc
    return yaml


def _require(mapping: dict, key: str, ctx: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{ctx}: missing required key {key!r}.")
    return mapping[key]


def _as_float(value: Any, ctx: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{ctx}: expected a number, got {value!r}.") from exc


def _as_int(value: Any, ctx: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{ctx}: expected an integer, got {value!r}.") from exc


def _fortran_half_order(fd_order: int, npml: int) -> int:
    """Convert full FD order (``fd_order``) to half-order for stencil loops.

    Mirrors Fortran ``Input.f90``: ``norder = int(norder/2)`` then
    clamped so that ``2*norder <= npml + 1``.
    """
    n = fd_order // 2
    if 2 * n > npml + 1:
        n = (npml + 1) // 2
    return n


def _resolve_path(base_dir: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _parse_grid(raw: dict, ctx: str = "grid") -> dict[str, float | int]:
    """Return nx, nz, nt and spacing/origin fields."""
    if "nx" in raw and "nz" in raw:
        nx = _as_int(raw["nx"], f"{ctx}.nx")
        nz = _as_int(raw["nz"], f"{ctx}.nz")
        x0 = _as_float(raw.get("x0", 0.0), f"{ctx}.x0")
        z0 = _as_float(raw.get("z0", 0.0), f"{ctx}.z0")
        dx = _as_float(_require(raw, "dx", ctx), f"{ctx}.dx")
        dz = _as_float(_require(raw, "dz", ctx), f"{ctx}.dz")
        xn = x0 + (nx - 1) * dx
        zn = z0 + (nz - 1) * dz
    else:
        x0 = _as_float(_require(raw, "x0", ctx), f"{ctx}.x0")
        xn = _as_float(_require(raw, "xn", ctx), f"{ctx}.xn")
        z0 = _as_float(_require(raw, "z0", ctx), f"{ctx}.z0")
        zn = _as_float(_require(raw, "zn", ctx), f"{ctx}.zn")
        dx = _as_float(_require(raw, "dx", ctx), f"{ctx}.dx")
        dz = _as_float(_require(raw, "dz", ctx), f"{ctx}.dz")
        nx = int(round((xn - x0) / dx)) + 1
        nz = int(round((zn - z0) / dz)) + 1

    dt = _as_float(_require(raw, "dt", ctx), f"{ctx}.dt")
    if dt <= 0:
        raise ValueError(f"{ctx}.dt must be positive.")

    if "nt" in raw:
        nt = _as_int(raw["nt"], f"{ctx}.nt")
        tmax = (nt - 1) * dt
    else:
        tmax = _as_float(_require(raw, "tmax", ctx), f"{ctx}.tmax")
        nt = int(round(tmax / dt)) + 1

    if nx < 2 or nz < 2 or nt < 2:
        raise ValueError(f"{ctx}: nx, nz, and nt must each be >= 2.")

    return {
        "x0": x0,
        "xn": xn,
        "z0": z0,
        "zn": zn,
        "dx": dx,
        "dz": dz,
        "dt": dt,
        "tmax": tmax,
        "nx": nx,
        "nz": nz,
        "nt": nt,
    }


def _parse_fd(raw: dict | None, npml_default: int = 20) -> dict[str, Any]:
    raw = raw or {}
    ctx = "fd"
    npml = _as_int(raw.get("npml", npml_default), f"{ctx}.npml")

    if "norder_half" in raw:
        norder_half = _as_int(raw["norder_half"], f"{ctx}.norder_half")
    elif "norder_full" in raw:
        fd_order = _as_int(raw["norder_full"], f"{ctx}.norder_full")
        norder_half = _fortran_half_order(fd_order, npml)
    elif "fd_order" in raw:
        fd_order = _as_int(raw["fd_order"], f"{ctx}.fd_order")
        norder_half = _fortran_half_order(fd_order, npml)
    else:
        norder_half = 3

    return {
        "norder_half": norder_half,
        "npml": npml,
        "is_pml_top": bool(raw.get("is_pml_top", False)),
        "outsnap": bool(raw.get("outsnap", False)),
        "nstep_snapshot": _as_int(raw.get("nstep_snapshot", 100), f"{ctx}.nstep_snapshot"),
        "kappa_max": _as_float(raw.get("kappa_max", 1.0), f"{ctx}.kappa_max"),
        "alpha_max": _as_float(raw.get("alpha_max", 0.0), f"{ctx}.alpha_max"),
        "cpml_npower": _as_int(raw.get("cpml_npower", 2), f"{ctx}.cpml_npower"),
    }


def _parse_reference(raw: dict | None) -> dict[str, float]:
    raw = raw or {}
    return {
        "lat_org": _as_float(raw.get("lat_org", 0.0), "reference.lat_org"),
        "lon_org": _as_float(raw.get("lon_org", 0.0), "reference.lon_org"),
        "az_profile": _as_float(raw.get("az_profile", 0.0), "reference.az_profile"),
    }


def _parse_source(raw: dict) -> tuple[RickerSource, float, float | None, float | None]:
    ctx = "source"
    f0 = _as_float(_require(raw, "f0", ctx), f"{ctx}.f0")
    strength = _as_float(raw.get("strength", 1.0), f"{ctx}.strength")
    incidence = str(_require(raw, "incidence", ctx)).upper()
    t0 = raw.get("t0")
    t0_f = None if t0 is None else _as_float(t0, f"{ctx}.t0")

    if "rayp_deg" in raw:
        rayp_deg = _as_float(raw["rayp_deg"], f"{ctx}.rayp_deg")
    elif "p_si" in raw:
        p_si = _as_float(raw["p_si"], f"{ctx}.p_si")
        rayp_deg = p_si * np.pi * 6371.0e3 / 180.0
    else:
        raise ValueError(f"{ctx}: provide either 'rayp_deg' or 'p_si'.")

    event = raw.get("event", {})
    evla = event.get("lat")
    evlo = event.get("lon")
    evla_f = None if evla is None else _as_float(evla, f"{ctx}.event.lat")
    evlo_f = None if evlo is None else _as_float(evlo, f"{ctx}.event.lon")

    source = RickerSource(
        f0=f0,
        strength=strength,
        incidence=incidence,
        t0=t0_f,
    )
    return source, rayp_deg, evla_f, evlo_f


def _parse_geometry(
    raw: dict | None,
    *,
    rayp_deg: float,
    evla: float | None,
    evlo: float | None,
    reference: dict[str, float],
    zn: float,
    xn: float,
    x0: float,
    model_left: FKLayerModel,
) -> tuple[float, float | None]:
    raw = raw or {}
    ctx = "geometry"

    if "phi_deg" in raw:
        phi = np.deg2rad(_as_float(raw["phi_deg"], f"{ctx}.phi_deg"))
    elif "phi" in raw:
        phi = _as_float(raw["phi"], f"{ctx}.phi")
    else:
        if evla is None or evlo is None:
            raise ValueError(
                f"{ctx}: set geometry.phi_deg (or phi), or provide "
                "source.event.lat/lon plus reference.lat_org/lon_org."
            )
        baz = backazimuth(evla, evlo, reference["lat_org"], reference["lon_org"])
        phi = np.deg2rad(baz - reference["az_profile"])

    z_init_raw = raw.get("z_init")
    if z_init_raw is None:
        z_init = None
    else:
        z_init = _as_float(z_init_raw, f"{ctx}.z_init")

    if z_init is None and raw.get("auto_z_init", True):
        p_si = rayp_deg_to_si(rayp_deg)
        vp_half = model_left.vp[model_left.halfspace_index]
        profile_length = xn - x0
        z_init = init_depth(p_si, vp_half, profile_length, zn)

    return phi, z_init


def _parse_layers(raw_layers: list, ctx: str) -> FKLayerModel:
    if not raw_layers:
        raise ValueError(f"{ctx}: at least one layer entry is required.")
    vp, vs, z_top, rho = [], [], [], []
    has_rho = False
    for i, layer in enumerate(raw_layers):
        lctx = f"{ctx}[{i}]"
        if not isinstance(layer, dict):
            raise ValueError(f"{lctx}: each layer must be a mapping.")
        vp.append(_as_float(_require(layer, "vp", lctx), f"{lctx}.vp"))
        vs.append(_as_float(_require(layer, "vs", lctx), f"{lctx}.vs"))
        z_top.append(_as_float(_require(layer, "z_top", lctx), f"{lctx}.z_top"))
        if "rho" in layer:
            has_rho = True
            rho.append(_as_float(layer["rho"], f"{lctx}.rho"))
    rho_arr = np.array(rho) if has_rho else None
    return FKLayerModel.from_layers(vp, vs, z_top, rho=rho_arr)


def _parse_fk_model_side(
    raw: Any,
    *,
    base_dir: Path,
    ctx: str,
    fallback: FKLayerModel | None = None,
) -> FKLayerModel:
    if raw is None:
        if fallback is None:
            raise ValueError(f"{ctx}: FK model is required.")
        return fallback
    if raw == "same_as_left":
        if fallback is None:
            raise ValueError(f"{ctx}: 'same_as_left' requires fk_left to be defined first.")
        return fallback
    if isinstance(raw, str):
        return read_fk_model(_resolve_path(base_dir, raw))
    if isinstance(raw, dict):
        if "file" in raw:
            return read_fk_model(_resolve_path(base_dir, raw["file"]))
        if "layers" in raw:
            return _parse_layers(raw["layers"], f"{ctx}.layers")
    raise ValueError(
        f"{ctx}: expected a file path, {{file: ...}}, {{layers: [...]}}, "
        "or 'same_as_left'."
    )


def _parse_receivers(raw: dict, *, grid: dict[str, float | int]) -> tuple[np.ndarray, np.ndarray]:
    ctx = "receivers"
    rtype = str(raw.get("type", "uniform")).lower()

    if rtype == "uniform":
        x0 = _as_float(raw.get("x0", grid["x0"]), f"{ctx}.x0")
        xn = _as_float(raw.get("xn", grid["xn"]), f"{ctx}.xn")
        count = _as_int(_require(raw, "count", ctx), f"{ctx}.count")
        z = _as_float(raw.get("z", 0.0), f"{ctx}.z")
        rx = np.linspace(x0, xn, count)
        rz = np.full(count, z, dtype=float)
        return rx, rz

    if rtype == "linspace":
        x0 = _as_float(_require(raw, "x0", ctx), f"{ctx}.x0")
        x1 = _as_float(_require(raw, "x1", ctx), f"{ctx}.x1")
        count = _as_int(_require(raw, "count", ctx), f"{ctx}.count")
        z = _as_float(raw.get("z", 0.0), f"{ctx}.z")
        rx = np.linspace(x0, x1, count)
        rz = np.full(count, z, dtype=float)
        return rx, rz

    if rtype == "list":
        stations = _require(raw, "stations", ctx)
        if not stations:
            raise ValueError(f"{ctx}.stations must be non-empty.")
        rx, rz = [], []
        for i, st in enumerate(stations):
            sctx = f"{ctx}.stations[{i}]"
            if isinstance(st, (list, tuple)) and len(st) >= 2:
                rx.append(_as_float(st[0], f"{sctx}[0]"))
                rz.append(_as_float(st[1], f"{sctx}[1]"))
            elif isinstance(st, dict):
                rx.append(_as_float(_require(st, "x", sctx), f"{sctx}.x"))
                rz.append(_as_float(st.get("z", 0.0), f"{sctx}.z"))
            else:
                raise ValueError(
                    f"{sctx}: each station must be [x, z] or {{x:, z:}}."
                )
        return np.asarray(rx, dtype=float), np.asarray(rz, dtype=float)

    if rtype == "file":
        from .io_fortran import read_receivers

        path = _resolve_path(Path(raw.get("_base_dir", ".")), _require(raw, "path", ctx))
        return read_receivers(path)

    raise ValueError(
        f"{ctx}.type must be one of: uniform, linspace, list, file."
    )


def _parse_fd_2d(raw: dict | None, *, base_dir: Path) -> dict[str, Any] | None:
    if raw is None:
        return None
    ctx = "fd_2d"
    model_type = str(raw.get("type", "from_fk")).lower()
    if model_type == "from_fk":
        return {"type": "from_fk"}
    if model_type == "su":
        return {
            "type": "su",
            "vp": _resolve_path(base_dir, _require(raw, "vp", ctx)),
            "vs": _resolve_path(base_dir, _require(raw, "vs", ctx)),
        }
    raise ValueError(f"{ctx}.type must be 'from_fk' or 'su'.")


@dataclass
class YamlSimulation:
    """Fully parsed simulation configuration from a YAML file."""

    name: str
    mode: Literal["fk", "hybrid"]
    config: HybridConfig
    model_left: FKLayerModel
    model_right: FKLayerModel
    source: RickerSource
    rayp_deg: float
    receivers_x: np.ndarray
    receivers_z: np.ndarray
    phi: float
    z_init: float | None
    x0_plane: float
    lat_org: float
    lon_org: float
    az_profile: float
    fk_method: str = "kennett_numba"
    freq_damping: float = 0.0
    fd_2d: dict[str, Any] | None = None
    output_dir: Path | None = None
    output_format: str = "su"
    snapshot_interval: int = 0
    yaml_path: Path | None = field(default=None, repr=False)

    @property
    def p_si(self) -> float:
        return rayp_deg_to_si(self.rayp_deg)

    @property
    def dt(self) -> float:
        return self.config.dt

    @property
    def nt(self) -> int:
        return self.config.nt


def load_yaml(path: str | Path) -> YamlSimulation:
    """Load a simulation configuration from a YAML file.

    Relative paths inside the YAML (``file:``, ``fd_2d.vp``, etc.) are resolved
    relative to the YAML file's directory.
    """
    yaml = _import_yaml()
    yaml_path = Path(path).resolve()
    base_dir = yaml_path.parent

    with yaml_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: root element must be a mapping.")

    name = str(raw.get("name", yaml_path.stem))
    mode = str(raw.get("mode", "fk")).lower()
    if mode not in ("fk", "hybrid"):
        raise ValueError(f"{yaml_path}: mode must be 'fk' or 'hybrid'.")

    grid = _parse_grid(_require(raw, "grid", "root"))
    fd = _parse_fd(raw.get("fd"))
    reference = _parse_reference(raw.get("reference"))
    source, rayp_deg, evla, evlo = _parse_source(_require(raw, "source", "root"))

    models_raw = _require(raw, "models", "root")
    model_left = _parse_fk_model_side(
        models_raw.get("fk_left"),
        base_dir=base_dir,
        ctx="models.fk_left",
    )
    model_right = _parse_fk_model_side(
        models_raw.get("fk_right", "same_as_left"),
        base_dir=base_dir,
        ctx="models.fk_right",
        fallback=model_left,
    )

    receivers_raw = _require(raw, "receivers", "root")
    if receivers_raw.get("type") == "file":
        receivers_raw = {**receivers_raw, "_base_dir": str(base_dir)}
    receivers_x, receivers_z = _parse_receivers(receivers_raw, grid=grid)

    phi, z_init = _parse_geometry(
        raw.get("geometry"),
        rayp_deg=rayp_deg,
        evla=evla,
        evlo=evlo,
        reference=reference,
        zn=float(grid["zn"]),
        xn=float(grid["xn"]),
        x0=float(grid["x0"]),
        model_left=model_left,
    )

    config = HybridConfig(
        nx=int(grid["nx"]),
        nz=int(grid["nz"]),
        dx=float(grid["dx"]),
        dz=float(grid["dz"]),
        x0=float(grid["x0"]),
        z0=float(grid["z0"]),
        dt=float(grid["dt"]),
        nt=int(grid["nt"]),
        norder=int(fd["norder_half"]),
        npml=int(fd["npml"]),
        kappa_max=float(fd["kappa_max"]),
        alpha_max=float(fd["alpha_max"]),
        cpml_npower=int(fd["cpml_npower"]),
    )

    fk_raw = raw.get("fk", {})
    fk_method = str(fk_raw.get("method", "kennett_numba"))
    freq_damping = _as_float(fk_raw.get("freq_damping", 0.0), "fk.freq_damping")

    fd_2d = _parse_fd_2d(raw.get("fd_2d"), base_dir=base_dir) if mode == "hybrid" else None

    output_raw = raw.get("output", {})
    output_dir = output_raw.get("directory")
    output_dir_path = None if output_dir is None else Path(str(output_dir))
    output_format = str(output_raw.get("format", "su")).lower()
    snapshot_interval = _as_int(output_raw.get("snapshot_interval", 0), "output.snapshot_interval")

    return YamlSimulation(
        name=name,
        mode=mode,  # type: ignore[arg-type]
        config=config,
        model_left=model_left,
        model_right=model_right,
        source=source,
        rayp_deg=rayp_deg,
        receivers_x=receivers_x,
        receivers_z=receivers_z,
        phi=phi,
        z_init=z_init,
        x0_plane=float(grid["x0"]),
        lat_org=reference["lat_org"],
        lon_org=reference["lon_org"],
        az_profile=reference["az_profile"],
        fk_method=fk_method,
        freq_damping=freq_damping,
        fd_2d=fd_2d,
        output_dir=output_dir_path,
        output_format=output_format,
        snapshot_interval=snapshot_interval,
        yaml_path=yaml_path,
    )
