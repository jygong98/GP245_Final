#!/usr/bin/env python3
"""Run the Python hybrid FDFK2D benchmark for one case directory."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
FDFK_PYTHON = Path(os.environ.get("FDFK_PYTHON", BENCH_DIR.parent.parent)).resolve()
sys.path.insert(0, str(FDFK_PYTHON))

from fk.geometry import rayp_deg_to_si  # noqa: E402
from fk.io_fortran import (  # noqa: E402
    load_example,
    read_fk_model,
    read_su,
    write_su,
)
from hybrid.config import HybridConfig  # noqa: E402
from hybrid.engine import compute_hybrid_seismograms  # noqa: E402


def fortran_half_order(raw_order: int, npml: int) -> int:
    """Match Input.f90 norder conversion (full order in FD_model -> half-order)."""
    n = raw_order // 2
    if 2 * n > npml + 1:
        n = (npml + 1) // 2
    return n


def _model_stems(inpar_path: Path) -> tuple[str, str]:
    lines = [ln.strip() for ln in inpar_path.read_text().splitlines() if ln.strip()]
    if len(lines) < 7:
        raise ValueError(f"expected 7 lines in {inpar_path}")
    return lines[5], lines[6]


def _without_pml_path(input_dir: Path, stem: str) -> Path:
    if stem.endswith(".su"):
        stem = stem[:-3]
    path = input_dir / f"{stem}_without_pml.su"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"missing {path.name}; run Fortran once or build_benchmark_cases.py"
    )


def main() -> None:
    input_dir = BENCH_DIR / "input"
    out_dir = BENCH_DIR / "output" / "python"
    bench_dir = BENCH_DIR / "output" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    ex = load_example(input_dir, side="left")
    fd = ex.fd
    nx = int(round((fd.xn - fd.x0) / fd.dx)) + 1
    nz = int(round((fd.zn - fd.z0) / fd.dz)) + 1

    config = HybridConfig(
        nx=nx,
        nz=nz,
        dx=fd.dx,
        dz=fd.dz,
        x0=fd.x0,
        z0=fd.z0,
        dt=fd.dt,
        nt=fd.nt,
        norder=fd.norder,
        npml=fd.npml,
    )

    vp_stem, vs_stem = _model_stems(input_dir / "inpar.dat")
    vp_tr, _ = read_su(_without_pml_path(input_dir, vp_stem))
    vs_tr, _ = read_su(_without_pml_path(input_dir, vs_stem))
    vp_2d = vp_tr.T.astype(np.float64)
    vs_2d = vs_tr.T.astype(np.float64)
    rho_2d = (vp_2d + 980.0) / 2.760

    model_left = read_fk_model(input_dir / "FK_model_left.dat")
    model_right = read_fk_model(input_dir / "FK_model_right.dat")
    p_si = rayp_deg_to_si(ex.src.rayp_deg)

    vp_half = model_left.vp[model_left.halfspace_index]
    sin_arg = min(1.0, max(-1.0, p_si * vp_half))
    z_init = fd.zn + 0.5 * (nx - 1) * fd.dx * np.tan(np.arcsin(sin_arg))

    print(
        f"[Python benchmark] case={BENCH_DIR.name} "
        f"grid=({nx}x{nz}) nt={fd.nt} nrcv={ex.rx.size} norder={config.norder}"
    )

    t0 = time.perf_counter()
    result = compute_hybrid_seismograms(
        config,
        model_left,
        model_right,
        ex.source,
        p_si,
        receivers_x=ex.rx,
        receivers_z=ex.rz,
        vp_2d=vp_2d,
        vs_2d=vs_2d,
        rho_2d=rho_2d,
        phi=ex.phi,
        z_init=z_init,
        bwf_pad_factor=2.0,  # anti-wraparound for 85 s record (nt=8501)
        verbose=True,
    )
    wall_sec = time.perf_counter() - t0

    write_su(out_dir / "seisx.su", result.ux, fd.dt)
    write_su(out_dir / "seisz.su", result.uz, fd.dt)

    timing = {
        "case": BENCH_DIR.name,
        "python_wall_sec": wall_sec,
        "nx": nx,
        "nz": nz,
        "nt": fd.nt,
        "nrcv": int(ex.rx.size),
        "norder": config.norder,
        "npml": fd.npml,
        "dt": fd.dt,
    }
    timing_path = bench_dir / "python.timing.json"
    timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    print(f"PYTHON_WALL_SEC={wall_sec:.3f}")
    print(f"Wrote {timing_path}")


if __name__ == "__main__":
    main()
