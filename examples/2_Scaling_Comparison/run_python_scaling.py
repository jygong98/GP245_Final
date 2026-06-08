"""Run Python hybrid FDFK scaling benchmarks and cache wall-clock times.

Thread scaling: baseline grid (1001×251), 1–12 Numba/OpenMP threads, tmax=40 s.
Spatial scaling: Fortran-matched grids at 8 threads, scale factors 1, 2, 4, 8, 16.

Results are written to ``python_scaling_times.json`` beside this file.
Each case can also be executed in an isolated subprocess (recommended for thread
scaling so Numba picks up ``NUMBA_NUM_THREADS`` before JIT compilation).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
INPUT_DIR = PROJECT_ROOT / "benchmark" / "test1_P_out" / "input"
CACHE_PATH = EXAMPLE_DIR / "python_scaling_times.json"

# Match Fortran scaling grids (nx, nz) from FDFK_Scaling_Fortran logs.
BASELINE_NX, BASELINE_NZ = 1001, 251
SPATIAL_GRIDS: dict[int, tuple[int, int]] = {
    1: (1001, 251),
    2: (1001, 501),
    4: (2001, 501),
    8: (2001, 1001),
    16: (4001, 1001),
}

# FD_model.dat baseline parameters (test1).
DX = DZ = 200.0
DT = 0.01
TMAX = 40.0
NT = int(round(TMAX / DT)) + 1  # 4001
X0 = Z0 = 0.0
NPML = 20
NORDER = 8  # half-order for fd_order=16 with npml=20


@dataclass
class PythonScalingRun:
    """One timed Python hybrid simulation."""

    suite: str  # "thread" or "spatial"
    label: str
    threads: int
    spatial_factor: int | None
    nx: int
    nz: int
    nt: int
    tmax: float
    grid_points: int
    time_seconds: float
    complete: bool
    host: str
    timestamp: str


def _configure_threads(n_threads: int) -> None:
    os.environ["NUMBA_NUM_THREADS"] = str(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _project_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def _receivers_for_grid(nx: int, dx: float, x0: float) -> tuple[np.ndarray, np.ndarray]:
    """Surface receivers along the profile (401 traces, same spacing as test1)."""
    xn = x0 + (nx - 1) * dx
    rx = np.linspace(x0, xn, num=min(401, nx), dtype=np.float64)
    rz = np.zeros_like(rx)
    return rx, rz


def run_hybrid_simulation(
    *,
    nx: int,
    nz: int,
    n_threads: int,
    nt: int = NT,
    verbose: bool = False,
) -> float:
    """Run one hybrid simulation; return wall-clock seconds."""
    _configure_threads(n_threads)
    _project_path()

    from numba import set_num_threads

    set_num_threads(n_threads)

    from fk.geometry import rayp_deg_to_si
    from fk.io_fortran import load_example, read_fk_model
    from hybrid.config import HybridConfig
    from hybrid.engine import compute_hybrid_seismograms

    ex = load_example(INPUT_DIR, side="left")
    zn = Z0 + (nz - 1) * DZ

    config = HybridConfig(
        nx=nx,
        nz=nz,
        dx=DX,
        dz=DZ,
        x0=X0,
        z0=Z0,
        dt=DT,
        nt=nt,
        norder=NORDER,
        npml=NPML,
    )

    model_left = read_fk_model(INPUT_DIR / "FK_model_left.dat")
    model_right = read_fk_model(INPUT_DIR / "FK_model_right.dat")
    p_si = rayp_deg_to_si(ex.src.rayp_deg)

    if ex.source.incidence == "P":
        v_half = model_left.vp[model_left.halfspace_index]
    else:
        v_half = model_left.vs[model_left.halfspace_index]
    sin_arg = min(1.0, max(-1.0, p_si * v_half))
    z_init = zn + 0.5 * (nx - 1) * DX * np.tan(np.arcsin(sin_arg))

    rx, rz = _receivers_for_grid(nx, DX, X0)

    t0 = time.perf_counter()
    compute_hybrid_seismograms(
        config,
        model_left,
        model_right,
        ex.source,
        p_si,
        receivers_x=rx,
        receivers_z=rz,
        phi=ex.phi,
        z_init=z_init,
        verbose=verbose,
    )
    return time.perf_counter() - t0


def _run_in_subprocess(
    *,
    nx: int,
    nz: int,
    n_threads: int,
    nt: int,
    verbose: bool,
) -> float:
    """Spawn a fresh Python process for one case (clean Numba thread pool)."""
    env = os.environ.copy()
    env["NUMBA_NUM_THREADS"] = str(n_threads)
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--nx",
        str(nx),
        "--nz",
        str(nz),
        "--thread-count",
        str(n_threads),
        "--nt",
        str(nt),
    ]
    if verbose:
        cmd.append("--verbose")

    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(EXAMPLE_DIR),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (threads={n_threads}, grid={nx}x{nz}):\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return float(lines[-1])


def _host_label() -> str:
    return platform.processor() or platform.machine() or "unknown"


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {"runs": [], "metadata": {}}


def _cached_labels(*, suite: str) -> set[str]:
    return {
        r["label"]
        for r in _load_cache().get("runs", [])
        if r.get("suite") == suite and r.get("label")
    }


def missing_thread_labels(threads: list[int] | None = None) -> list[str]:
    """Thread-scaling labels not yet present in the cache file."""
    if threads is None:
        threads = list(range(1, 13))
    existing = _cached_labels(suite="thread")
    return [f"python_t{n}" for n in threads if f"python_t{n}" not in existing]


def missing_spatial_labels(scales: list[int] | None = None) -> list[str]:
    """Spatial-scaling labels not yet present in the cache file."""
    if scales is None:
        scales = [1, 2, 4, 8, 16]
    existing = _cached_labels(suite="spatial")
    return [f"python_s{factor}" for factor in scales if f"python_s{factor}" not in existing]


def _save_cache(payload: dict[str, Any]) -> None:
    CACHE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _upsert_run(run: PythonScalingRun, *, metadata: dict[str, Any]) -> None:
    """Persist one completed run immediately (safe if the suite is interrupted)."""
    cache = _load_cache()
    runs = [r for r in cache.get("runs", []) if r.get("label") != run.label]
    runs.append(asdict(run))
    cache["runs"] = runs
    cache["metadata"] = {**cache.get("metadata", {}), **metadata}
    _save_cache(cache)


def _run_record(
    *,
    suite: str,
    label: str,
    threads: int,
    spatial_factor: int | None,
    nx: int,
    nz: int,
    nt: int,
    wall_sec: float,
) -> PythonScalingRun:
    tmax = (nt - 1) * DT
    return PythonScalingRun(
        suite=suite,
        label=label,
        threads=threads,
        spatial_factor=spatial_factor,
        nx=nx,
        nz=nz,
        nt=nt,
        tmax=tmax,
        grid_points=nx * nz,
        time_seconds=wall_sec,
        complete=tmax >= TMAX - 1e-9,
        host=_host_label(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_thread_scaling(
    *,
    threads: list[int] | None = None,
    use_subprocess: bool = True,
    verbose: bool = True,
    skip_existing: bool = True,
) -> list[PythonScalingRun]:
    """Time baseline grid for each thread count."""
    if threads is None:
        threads = list(range(1, 13))

    cache = _load_cache()
    existing = {
        (r["suite"], r["label"]): r
        for r in cache.get("runs", [])
        if r.get("suite") == "thread"
    }

    new_runs: list[PythonScalingRun] = []
    runner = _run_in_subprocess if use_subprocess else (
        lambda **kw: run_hybrid_simulation(**kw, verbose=verbose)
    )

    for n in threads:
        label = f"python_t{n}"
        if skip_existing and label in existing:
            if verbose:
                print(f"[skip] {label} cached ({existing[label]['time_seconds']:.1f}s)")
            continue

        if verbose:
            print(f"[thread] {n} threads on {BASELINE_NX}x{BASELINE_NZ}, nt={NT} ...", flush=True)
        wall = runner(nx=BASELINE_NX, nz=BASELINE_NZ, n_threads=n, nt=NT, verbose=verbose)
        rec = _run_record(
            suite="thread",
            label=label,
            threads=n,
            spatial_factor=None,
            nx=BASELINE_NX,
            nz=BASELINE_NZ,
            nt=NT,
            wall_sec=wall,
        )
        new_runs.append(rec)
        _upsert_run(
            rec,
            metadata={
                "input_dir": str(INPUT_DIR),
                "baseline_grid": [BASELINE_NX, BASELINE_NZ],
                "tmax": TMAX,
                "updated": datetime.now(timezone.utc).isoformat(),
            },
        )
        if verbose:
            print(f"  -> {wall:.1f}s")

    return new_runs


def run_spatial_scaling(
    *,
    scales: list[int] | None = None,
    n_threads: int = 8,
    use_subprocess: bool = True,
    verbose: bool = True,
    skip_existing: bool = True,
) -> list[PythonScalingRun]:
    """Time spatial grids at fixed thread count."""
    if scales is None:
        scales = [1, 2, 4, 8, 16]

    cache = _load_cache()
    existing = {
        (r["suite"], r["label"]): r
        for r in cache.get("runs", [])
        if r.get("suite") == "spatial"
    }

    new_runs: list[PythonScalingRun] = []
    runner = _run_in_subprocess if use_subprocess else (
        lambda **kw: run_hybrid_simulation(**kw, verbose=verbose)
    )

    for factor in scales:
        nx, nz = SPATIAL_GRIDS[factor]
        label = f"python_s{factor}"
        if skip_existing and label in existing:
            if verbose:
                print(f"[skip] {label} cached ({existing[label]['time_seconds']:.1f}s)")
            continue

        if verbose:
            print(
                f"[spatial] scale={factor} ({nx}x{nz}), {n_threads} threads, nt={NT} ...",
                flush=True,
            )
        wall = runner(nx=nx, nz=nz, n_threads=n_threads, nt=NT, verbose=verbose)
        rec = _run_record(
            suite="spatial",
            label=label,
            threads=n_threads,
            spatial_factor=factor,
            nx=nx,
            nz=nz,
            nt=NT,
            wall_sec=wall,
        )
        new_runs.append(rec)
        _upsert_run(
            rec,
            metadata={
                "input_dir": str(INPUT_DIR),
                "spatial_grids": {str(k): list(v) for k, v in SPATIAL_GRIDS.items()},
                "tmax": TMAX,
                "spatial_threads": n_threads,
                "updated": datetime.now(timezone.utc).isoformat(),
            },
        )
        if verbose:
            print(f"  -> {wall:.1f}s")

    return new_runs


def format_python_table(runs: list[dict[str, Any]], *, key: str) -> str:
    """Plain-text summary for thread or spatial runs."""
    if key == "threads":
        selected = [r for r in runs if r.get("suite") == "thread"]
        selected.sort(key=lambda r: r["threads"])
        header = f"{'Label':<14} {'Threads':>8} {'Time (s)':>10} {'Complete':>9}"
    else:
        selected = [r for r in runs if r.get("suite") == "spatial"]
        selected.sort(key=lambda r: r.get("spatial_factor") or 0)
        header = (
            f"{'Label':<14} {'Scale':>6} {'nx':>6} {'nz':>6} {'Grid pts':>10} "
            f"{'Time (s)':>10} {'Complete':>9}"
        )

    lines = [header, "-" * len(header)]
    for run in selected:
        if key == "threads":
            lines.append(
                f"{run['label']:<14} {run['threads']:>8} {run['time_seconds']:>10.1f} "
                f"{str(run['complete']):>9}"
            )
        else:
            lines.append(
                f"{run['label']:<14} {run['spatial_factor']:>6} {run['nx']:>6} {run['nz']:>6} "
                f"{run['grid_points']:>10} {run['time_seconds']:>10.1f} {str(run['complete']):>9}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Python FDFK scaling benchmarks")
    parser.add_argument("--worker", action="store_true",
                        help="Internal: run one simulation and print wall time")
    parser.add_argument("--nx", type=int, default=BASELINE_NX)
    parser.add_argument("--nz", type=int, default=BASELINE_NZ)
    parser.add_argument("--nt", type=int, default=NT)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--thread-count", type=int, default=1,
                        help="Worker mode: Numba thread count")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Worker mode: write single-run result JSON (avoids cache races)")
    parser.add_argument("--label", type=str, default=None,
                        help="Worker mode: run label (e.g. python_t8)")
    parser.add_argument("--worker-suite", choices=["thread", "spatial"], default=None,
                        help="Worker mode: scaling suite for --output-json")
    parser.add_argument("--spatial-factor", type=int, default=None,
                        help="Worker mode: spatial scale factor (spatial suite only)")
    parser.add_argument("--suite", choices=["thread", "spatial", "all"], default="all")
    parser.add_argument("--threads", type=int, nargs="*", default=None,
                        help="Thread counts for thread suite (default 1..12)")
    parser.add_argument("--spatial-scales", type=int, nargs="*", default=None,
                        help="Spatial scale factors (default 1 2 4 8 16)")
    parser.add_argument("--spatial-threads", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Re-run even if cached")
    parser.add_argument("--in-process", action="store_true",
                        help="Run in current process (faster startup, less reliable threads)")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if args.worker:
        wall = run_hybrid_simulation(
            nx=args.nx,
            nz=args.nz,
            n_threads=args.thread_count,
            nt=args.nt,
            verbose=args.verbose,
        )
        print(wall)
        if args.output_json is not None:
            if not args.label or not args.worker_suite:
                parser.error("--worker with --output-json requires --label and --worker-suite")
            rec = _run_record(
                suite=args.worker_suite,
                label=args.label,
                threads=args.thread_count,
                spatial_factor=args.spatial_factor,
                nx=args.nx,
                nz=args.nz,
                nt=args.nt,
                wall_sec=wall,
            )
            if args.worker_suite == "thread":
                metadata = {
                    "input_dir": str(INPUT_DIR),
                    "baseline_grid": [BASELINE_NX, BASELINE_NZ],
                    "tmax": TMAX,
                    "updated": datetime.now(timezone.utc).isoformat(),
                }
            else:
                metadata = {
                    "input_dir": str(INPUT_DIR),
                    "spatial_grids": {str(k): list(v) for k, v in SPATIAL_GRIDS.items()},
                    "tmax": TMAX,
                    "spatial_threads": args.thread_count,
                    "updated": datetime.now(timezone.utc).isoformat(),
                }
            payload = {"runs": [asdict(rec)], "metadata": metadata}
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
        return

    use_subprocess = not args.in_process
    verbose = not args.quiet
    skip_existing = not args.force

    if args.suite in ("thread", "all"):
        run_thread_scaling(
            threads=args.threads,
            use_subprocess=use_subprocess,
            verbose=verbose,
            skip_existing=skip_existing,
        )
    if args.suite in ("spatial", "all"):
        run_spatial_scaling(
            scales=args.spatial_scales,
            n_threads=args.spatial_threads,
            use_subprocess=use_subprocess,
            verbose=verbose,
            skip_existing=skip_existing,
        )

    cache = _load_cache()
    print("\nPython thread scaling")
    print(format_python_table(cache.get("runs", []), key="threads"))
    print("\nPython spatial scaling")
    print(format_python_table(cache.get("runs", []), key="spatial"))


if __name__ == "__main__":
    main()
