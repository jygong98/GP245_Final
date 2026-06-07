#!/usr/bin/env python3
"""Collect Fortran/Python timings and optional waveform checks."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fk.io_fortran import read_su  # noqa: E402


def _parse_fortran_time(path: Path) -> dict:
    if not path.exists():
        return {"fortran_wall_sec": None, "fortran_maxrss_kb": None}
    text = path.read_text()
    wall = None
    rss = None
    m = re.search(r"WALL_SEC=([0-9.]+)", text)
    if m:
        wall = float(m.group(1))
    m = re.search(r"MAXRSS_KB=([0-9]+)", text)
    if m:
        rss = int(m.group(1))
    return {"fortran_wall_sec": wall, "fortran_maxrss_kb": rss}


def _waveform_metrics(fortran_dir: Path, python_dir: Path) -> dict:
    metrics: dict = {}
    for comp in ("x", "z"):
        f_path = fortran_dir / f"seis{comp}.su"
        p_path = python_dir / f"seis{comp}.su"
        if not (f_path.exists() and p_path.exists()):
            metrics[f"uz_rmse" if comp == "z" else f"ux_rmse"] = None
            continue
        f_data, f_dt = read_su(f_path)
        p_data, p_dt = read_su(p_path)
        if f_data.shape != p_data.shape:
            metrics[f"u{comp}_shape_match"] = False
            metrics[f"u{comp}_rmse"] = None
            continue
        if abs(f_dt - p_dt) > 1e-9:
            metrics[f"u{comp}_dt_match"] = False
        diff = f_data.astype(np.float64) - p_data.astype(np.float64)
        rmse = float(np.sqrt(np.mean(diff**2)))
        denom = float(np.max(np.abs(f_data))) + 1e-30
        nrmse = rmse / denom
        cc = float(np.corrcoef(f_data.ravel(), p_data.ravel())[0, 1])
        metrics[f"u{comp}_rmse"] = rmse
        metrics[f"u{comp}_nrmse"] = nrmse
        metrics[f"u{comp}_cc"] = cc
    return metrics


def main() -> None:
    bench_out = BENCH_DIR / "output" / "benchmark"
    fortran_dir = BENCH_DIR / "output" / "fortran"
    python_dir = BENCH_DIR / "output" / "python"

    summary: dict = {"case": BENCH_DIR.name}
    summary.update(_parse_fortran_time(bench_out / "fortran.time"))

    py_path = bench_out / "python.timing.json"
    if py_path.exists():
        summary.update(json.loads(py_path.read_text()))
    else:
        summary["python_wall_sec"] = None

    f_wall = summary.get("fortran_wall_sec")
    p_wall = summary.get("python_wall_sec")
    if f_wall and p_wall and p_wall > 0:
        summary["speedup_fortran_over_python"] = p_wall / f_wall
        summary["speedup_python_over_fortran"] = f_wall / p_wall
    else:
        summary["speedup_fortran_over_python"] = None
        summary["speedup_python_over_fortran"] = None

    summary.update(_waveform_metrics(fortran_dir, python_dir))

    json_path = bench_out / "benchmark_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        f"FDFK2D benchmark summary — {BENCH_DIR.name}",
        "=" * 48,
        f"Fortran wall time (s): {f_wall!s}",
        f"Python wall time  (s): {p_wall!s}",
    ]
    if summary.get("speedup_fortran_over_python") is not None:
        lines.append(
            f"Speedup (Fortran faster): {summary['speedup_fortran_over_python']:.2f}x"
        )
        lines.append(
            f"Speedup (Python faster):  {summary['speedup_python_over_fortran']:.2f}x"
        )
    if summary.get("uz_rmse") is not None:
        lines.extend(
            [
                "",
                "Waveform check vs Fortran (uz preferred; ux may differ by rotation):",
                f"  uz RMSE: {summary['uz_rmse']:.3e}  NRMSE: {summary.get('uz_nrmse', 0):.3e}  CC: {summary.get('uz_cc', 0):.4f}",
            ]
        )
        if summary.get("ux_rmse") is not None:
            lines.append(
                f"  ux RMSE: {summary['ux_rmse']:.3e}  NRMSE: {summary.get('ux_nrmse', 0):.3e}  CC: {summary.get('ux_cc', 0):.4f}"
            )

    txt_path = bench_out / "benchmark_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {json_path} and {txt_path}")


if __name__ == "__main__":
    main()
