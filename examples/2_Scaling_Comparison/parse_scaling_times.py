"""Parse execution times from FDFK Fortran scaling output files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Matches progress lines such as:
# t =     0.00000 s  ... ;    0.367 min. remained
REMAINED_RE = re.compile(
    r"t\s*=\s*[\d.]+\s+s\s+[\d.E+-]+\s+[\d.E+-]+\s*;\s*([\d.]+)\s*min\.\s*remained",
    re.IGNORECASE,
)

# Matches Start/End time lines with variable spacing.
TIME_RE = re.compile(
    r"(?:Start|End)\s+time:\s*"
    r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*;\s*"
    r"(\d+)\s*:\s*(\d+)\s*:\s*(\d+)\s*:\s*(\d+)",
    re.IGNORECASE,
)

NX_NZ_RE = re.compile(r"nx\s*=\s*(\d+)\s*;\s*nz\s*=\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ScalingRun:
    """Parsed timing and grid metadata for one scaling run."""

    label: str
    path: Path
    threads: int | None
    spatial_factor: int | None
    nx: int | None
    nz: int | None
    grid_points: int | None
    time_seconds: float
    method: str
    complete: bool
    last_sim_time_s: float | None


def _parse_timestamp(match: re.Match[str]) -> datetime:
    year, month, day, hour, minute, second, millis = map(int, match.groups())
    return datetime(year, month, day, hour, minute, second, millis * 1000)


def _duration_from_timestamps(text: str) -> float | None:
    starts = [TIME_RE.search(line) for line in text.splitlines() if "start" in line.lower() and "time:" in line.lower()]
    ends = [TIME_RE.search(line) for line in text.splitlines() if "end" in line.lower() and "time:" in line.lower()]

    start_matches = [m for m in starts if m is not None]
    end_matches = [m for m in ends if m is not None]
    if not start_matches or not end_matches:
        return None

    start = _parse_timestamp(start_matches[0])
    end = _parse_timestamp(end_matches[-1])
    return (end - start).total_seconds()


def _duration_from_remained(text: str) -> float | None:
    remained = [float(m.group(1)) for m in REMAINED_RE.finditer(text)]
    if len(remained) < 2:
        return None
    return (remained[0] + remained[1]) * 60.0


def _last_sim_time(text: str) -> float | None:
    sim_times = re.findall(r"t\s*=\s*([\d.]+)\s+s", text)
    if not sim_times:
        return None
    return float(sim_times[-1])


def _parse_grid(text: str) -> tuple[int | None, int | None]:
    match = NX_NZ_RE.search(text)
    if match is None:
        return None, None
    nx, nz = map(int, match.groups())
    return nx, nz


def parse_out_file(path: Path, label: str, *, threads: int | None = None, spatial_factor: int | None = None) -> ScalingRun:
    """Extract wall-clock runtime from a Fortran *.out log file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    nx, nz = _parse_grid(text)
    grid_points = nx * nz if nx is not None and nz is not None else None
    last_t = _last_sim_time(text)
    complete = last_t is not None and last_t >= 39.99

    if _duration_from_timestamps(text) is not None:
        seconds = _duration_from_timestamps(text)
        method = "start_end"
    else:
        seconds = _duration_from_remained(text)
        method = "remained"

    if seconds is None:
        raise ValueError(f"Could not extract runtime from {path}")

    return ScalingRun(
        label=label,
        path=path,
        threads=threads,
        spatial_factor=spatial_factor,
        nx=nx,
        nz=nz,
        grid_points=grid_points,
        time_seconds=seconds,
        method=method,
        complete=complete,
        last_sim_time_s=last_t,
    )


def discover_runs(data_dir: Path) -> list[ScalingRun]:
    """Discover and parse all scaling runs under ``data_dir``."""
    runs: list[ScalingRun] = []

    thread_cases = [
        ("test1_1", 1),
        ("test1_2", 2),
        ("test1_4", 4),
        ("test1_8", 8),
        ("test1_16", 16),
        ("test1_64", 64),
    ]
    for dirname, n_threads in thread_cases:
        case_dir = data_dir / dirname
        out_files = sorted(case_dir.glob("*.out"))
        if not out_files:
            continue
        runs.append(parse_out_file(out_files[0], dirname, threads=n_threads))

    spatial_cases = [
        ("test1_8", 1),
        ("test1_s2", 2),
        ("test1_s4", 4),
        ("test1_s8", 8),
        ("test1_s16", 16),
        ("test1_s32", 32),
        ("test1_s64", 64),
    ]
    for dirname, factor in spatial_cases:
        if dirname == "test1_8":
            path = data_dir / dirname / "fdfk.out"
        else:
            path = data_dir / dirname / "fdfk1.out"
        if not path.exists():
            continue
        runs.append(parse_out_file(path, dirname, spatial_factor=factor))

    return runs


def format_summary_table(runs: list[ScalingRun], *, key: str) -> str:
    """Return a plain-text table for thread or spatial scaling runs."""
    if key == "threads":
        selected = [r for r in runs if r.threads is not None]
        selected.sort(key=lambda r: r.threads or 0)
        header = f"{'Case':<10} {'Threads':>8} {'Time (s)':>10} {'Method':>10} {'Complete':>9}"
    else:
        selected = [r for r in runs if r.spatial_factor is not None]
        selected.sort(key=lambda r: r.spatial_factor or 0)
        header = (
            f"{'Case':<10} {'Scale':>6} {'nx':>6} {'nz':>6} {'Grid pts':>10} "
            f"{'Time (s)':>10} {'Method':>10} {'Complete':>9}"
        )

    lines = [header, "-" * len(header)]
    for run in selected:
        if key == "threads":
            lines.append(
                f"{run.label:<10} {run.threads:>8} {run.time_seconds:>10.1f} "
                f"{run.method:>10} {str(run.complete):>9}"
            )
        else:
            lines.append(
                f"{run.label:<10} {run.spatial_factor:>6} {run.nx or 0:>6} {run.nz or 0:>6} "
                f"{run.grid_points or 0:>10} {run.time_seconds:>10.1f} {run.method:>10} {str(run.complete):>9}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    data_root = Path(__file__).resolve().parent / "FDFK_Scaling_Fortran"
    all_runs = discover_runs(data_root)
    print("Thread scaling")
    print(format_summary_table(all_runs, key="threads"))
    print()
    print("Spatial scaling")
    print(format_summary_table(all_runs, key="spatial"))
