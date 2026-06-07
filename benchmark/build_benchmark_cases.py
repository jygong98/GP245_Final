#!/usr/bin/env python3
"""Build self-contained HPC benchmark folders for test1 and Altyn.

Run once from the repository root (or benchmark/):

    python benchmark/build_benchmark_cases.py

Creates ``benchmark/test1/`` and ``benchmark/Altyn/`` with input decks,
SU velocity models (including PML padding required by FDFK2D), and job scripts.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
REF = ROOT / "ref_code" / "FDFK2D-master_experiment" / "examples"

_SU_HDR_BYTES = 240
_SU_NS_IDX = 57
_SU_DT_IDX = 58


def write_su(path: Path, data: np.ndarray, dz: float) -> None:
    """Write depth-domain SU (matches examples/input/writesu.m, domain=1)."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {data.shape}")
    ns, nx = data.shape
    # Depth-domain models: Fortran-written headers use 1000 (see Input.f90 head(59)=1e3).
    dt_us = 1000
    out = bytearray()
    # MATLAB writesu.m stores the header as uint16 (200 m spacing -> 200000 us).
    hdr = np.zeros(120, dtype="<u2")
    hdr[_SU_NS_IDX] = ns
    hdr[_SU_DT_IDX] = dt_us
    hdr_bytes = hdr.tobytes()
    for ix in range(nx):
        out += hdr_bytes
        out += data[:, ix].tobytes()
    path.write_bytes(bytes(out))


def read_su(path: Path) -> tuple[np.ndarray, float]:
    """Read SU depth-domain file -> array (ns, nx), dz."""
    raw = path.read_bytes()
    pos = 0
    cols: list[np.ndarray] = []
    dz = 1.0
    while pos + _SU_HDR_BYTES <= len(raw):
        hdr = np.frombuffer(raw[pos : pos + _SU_HDR_BYTES], dtype="<i2")
        ns = int(hdr[_SU_NS_IDX])
        pos += _SU_HDR_BYTES
        col = np.frombuffer(raw[pos : pos + ns * 4], dtype="<f4", count=ns)
        pos += ns * 4
        cols.append(col.copy())
        dz = hdr[_SU_DT_IDX] / 1e3
    if not cols:
        raise ValueError(f"empty or invalid SU file: {path}")
    return np.column_stack(cols), float(dz)


def pad_model_with_pml(model: np.ndarray, npml: int) -> np.ndarray:
    """Extend (nz_phys, nx_phys) model with PML columns and bottom rows."""
    nz_phys, nx_phys = model.shape
    bottom = np.repeat(model[-1:, :], npml, axis=0)
    model_z = np.vstack([model, bottom])
    left = np.repeat(model_z[:, :1], npml, axis=1)
    right = np.repeat(model_z[:, -1:], npml, axis=1)
    return np.hstack([left, model_z, right])


def strip_pml(model_pml: np.ndarray, npml: int) -> np.ndarray:
    """Strip PML padding -> (nz_phys, nx_phys)."""
    return model_pml[: model_pml.shape[0] - npml, npml:-npml]


def verify_case(case: str, nx_phys: int, nz_phys: int, npml: int, vp_name: str) -> None:
    """Sanity-check SU grid sizes against FD_model expectations."""
    inp = BENCH / case / "input"
    vp, _ = read_su(inp / vp_name)
    nz_exp = nz_phys + npml
    nx_exp = nx_phys + 2 * npml
    if vp.shape != (nz_exp, nx_exp):
        raise RuntimeError(f"{case}: {vp_name} shape {vp.shape} != ({nz_exp}, {nx_exp})")
    print(f"{case}: verified {vp_name} ({nz_exp}, {nx_exp})")


def build_test1() -> None:
    """One-layer model (examples/test1) with generated vp/vs SU files."""
    src = REF / "test1"
    dst = BENCH / "test1"
    inp = dst / "input"
    if inp.exists():
        shutil.rmtree(inp)
    inp.mkdir(parents=True)

    for name in (
        "FD_model.dat",
        "Source.dat",
        "Receiver.dat",
        "FK_model_left.dat",
        "FK_model_right.dat",
        "inpar.dat",
    ):
        shutil.copy2(src / "input" / name, inp / name)

    # Disable snapshots for fair timing (original saved as reference).
    fd_text = (inp / "FD_model.dat").read_text()
    (inp / "FD_model.with_snapshots.dat").write_text(fd_text)
    fd_text = fd_text.replace(".true.", ".false.", 1)
    (inp / "FD_model.dat").write_text(fd_text)

    npml = 20
    dx = dz = 200.0
    nx_phys = int(round(200_000 / dx)) + 1
    nz_phys = int(round(50_000 / dz)) + 1
    ih = int(round(35_000 / dz))

    vp = np.full((nz_phys, nx_phys), 6000.0, dtype=np.float32)
    vs = np.full((nz_phys, nx_phys), 3450.0, dtype=np.float32)
    vp[ih:, :] = 8000.0
    vs[ih:, :] = 4480.0

    vp_pml = pad_model_with_pml(vp, npml)
    vs_pml = pad_model_with_pml(vs, npml)
    write_su(inp / "vp_model.su", vp_pml, dz)
    write_su(inp / "vs_model.su", vs_pml, dz)
    write_su(inp / "vp_model_without_pml.su", vp, dz)
    write_su(inp / "vs_model_without_pml.su", vs, dz)

    print(f"test1: vp_model.su shape {vp_pml.shape} (nz, nx with PML)")
    verify_case("test1", nx_phys, nz_phys, npml, "vp_model.su")


def build_altyn() -> None:
    """Altyn Tagh profile; build PML-padded SU from shipped without_pml files."""
    src = REF / "Altyn"
    dst = BENCH / "Altyn"
    inp = dst / "input"
    if inp.exists():
        shutil.rmtree(inp)
    inp.mkdir(parents=True)

    for name in (
        "FD_model.dat",
        "Source.dat",
        "Receiver.dat",
        "FK_model_left.dat",
        "FK_model_right.dat",
        "inpar.dat",
    ):
        shutil.copy2(src / "input" / name, inp / name)

    fd_text = (inp / "FD_model.dat").read_text()
    (inp / "FD_model.with_snapshots.dat").write_text(fd_text)
    fd_text = fd_text.replace(".true.", ".false.", 1)
    (inp / "FD_model.dat").write_text(fd_text)

    npml = 20
    dz = 200.0
    vp_stem = "vp_model_smooth_2.4x1.2"
    vs_stem = "vs_model_smooth_2.4x1.2"

    vp_phys, _ = read_su(src / "input" / f"{vp_stem}_without_pml.su")
    vs_phys, _ = read_su(src / "input" / f"{vs_stem}_without_pml.su")
    if vp_phys.shape != vs_phys.shape:
        raise RuntimeError(f"Altyn vp/vs shape mismatch: {vp_phys.shape} vs {vs_phys.shape}")

    nx_phys = int(round(380_000 / 200.0)) + 1
    nz_phys = int(round(120_000 / 200.0)) + 1
    if vp_phys.shape != (nz_phys, nx_phys):
        raise RuntimeError(
            f"Altyn model shape {vp_phys.shape} != expected ({nz_phys}, {nx_phys})"
        )

    vp_pml = pad_model_with_pml(vp_phys.astype(np.float32), npml)
    vs_pml = pad_model_with_pml(vs_phys.astype(np.float32), npml)
    write_su(inp / f"{vp_stem}.su", vp_pml, dz)
    write_su(inp / f"{vs_stem}.su", vs_pml, dz)
    shutil.copy2(src / "input" / f"{vp_stem}_without_pml.su", inp / f"{vp_stem}_without_pml.su")
    shutil.copy2(src / "input" / f"{vs_stem}_without_pml.su", inp / f"{vs_stem}_without_pml.su")

    print(f"Altyn: {vp_stem}.su shape {vp_pml.shape} (nz, nx with PML)")
    verify_case("Altyn", nx_phys, nz_phys, npml, f"{vp_stem}.su")


def _deploy_scripts(case: str) -> None:
    """Copy shared run scripts into a case folder."""
    common = BENCH / "_common"
    dst = BENCH / case
    for name in ("run_job.sh", "run_python_benchmark.py", "summarize_benchmark.py"):
        shutil.copy2(common / name, dst / name)
        if name.endswith(".sh"):
            (dst / name).chmod(0o755)
    sbatch = dst / "submit.sbatch"
    if sbatch.exists():
        sbatch.chmod(0o755)


def main() -> None:
    build_test1()
    build_altyn()
    _deploy_scripts("test1")
    _deploy_scripts("Altyn")
    print("Benchmark cases ready: benchmark/test1, benchmark/Altyn")


if __name__ == "__main__":
    main()
