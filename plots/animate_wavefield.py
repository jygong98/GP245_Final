#!/usr/bin/env python3
"""Animate 2D wavefield snapshots from FDFK2D Fortran output.

Reads SU snapshot files produced by the Fortran FDFK2D code (when
``outsnap=.true.`` in FD_model.dat) and creates an MP4 or GIF animation
of the wavefield time evolution.

Snapshot file convention (from FDFK2D.f90):
    ./snapshots/{it:05d}ux.su   -- radial displacement at time step `it`
    ./snapshots/{it:05d}uz.su   -- vertical displacement at time step `it`

Each file contains `nx` traces of `nz` float32 samples (one trace per
x-grid column, with `nz` depth samples per trace).

Example
-------
    # Animate vertical component with FD_model.dat for axis metadata
    python plots/animate_wavefield.py ./snapshots --comp uz \\
        --fd-model input/FD_model.dat --out wavefield_uz.mp4

    # Quick GIF with lower quality
    python plots/animate_wavefield.py ./snapshots --comp ux \\
        --out wavefield_ux.gif --dpi 80 --fps 15

    # Just show a few frames without saving
    python plots/animate_wavefield.py ./snapshots --comp uz --show --nframes 10
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fk.io_fortran import read_traces
from plots.wavefield_snapshots import snapshot_for_plot


def parse_fd_model(path: Path) -> dict:
    """Parse FD_model.dat for grid parameters.

    Returns dict with keys: x0, xn, z0, zn, dx, dz, dt, nstep.
    """
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    values = [ln for ln in lines
              if ln[0] in "+-." or ln[0].isdigit()
              or ln.lower().startswith(".t") or ln.lower().startswith(".f")]

    def _float(s):
        return float(s.replace("d", "e").replace("D", "e"))

    g = values[1].split()
    x0, xn, z0, zn, dx, dz, dt = (_float(t) for t in g[:7])

    t_line = values[2].split()
    nstep_token = values[4].split()[1] if len(values) > 4 else "100"
    nstep = int(float(nstep_token))

    return dict(x0=x0, xn=xn, z0=z0, zn=zn, dx=dx, dz=dz, dt=dt, nstep=nstep)


def discover_snapshots(snap_dir: Path, comp: str) -> list[tuple[int, Path]]:
    """Find snapshot files and return sorted list of (timestep, path)."""
    pattern = str(snap_dir / f"*{comp}.su")
    files = glob.glob(pattern)
    results = []
    regex = re.compile(r"(\d{5})" + re.escape(comp) + r"\.su$")
    for f in files:
        m = regex.search(f)
        if m:
            results.append((int(m.group(1)), Path(f)))
    results.sort(key=lambda x: x[0])
    return results


def load_snapshot(path: Path) -> np.ndarray:
    """Load a single snapshot file -> 2D array (nx, nz)."""
    data, _ = read_traces(path)
    return data


def animate_wavefield(
    snap_dir: Path,
    comp: str = "uz",
    fd_model_path: Path | None = None,
    out_path: Path | None = None,
    *,
    fps: int = 20,
    dpi: int = 100,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "seismic",
    nframes: int | None = None,
    iz_plot_start: int = 0,
    interpolation: str = "nearest",
    show: bool = False,
) -> None:
    """Create wavefield animation from FDFK snapshot files.

    Parameters
    ----------
    snap_dir : Path
        Directory containing snapshot SU files.
    comp : str
        Component to animate: 'ux' or 'uz'.
    fd_model_path : Path, optional
        Path to FD_model.dat for axis metadata.
    out_path : Path, optional
        Output file (.mp4 or .gif). None = no save.
    fps : int
        Frames per second.
    dpi : int
        Output resolution.
    vmin, vmax : float, optional
        Color scale limits. If None, auto-determined from first frame.
    cmap : str
        Matplotlib colormap.
    nframes : int, optional
        Limit to first N frames (for quick previews).
    iz_plot_start : int
        Drop this many leading depth rows before plotting. Use ``1`` for Python
        hybrid snapshots (skip the free-surface ghost row at iz=0); keep ``0``
        for Fortran FDFK2D output where iz=1 is the physical surface.
    interpolation : str
        Matplotlib imshow interpolation. ``"nearest"`` avoids moiré on FD
        grid cells; ``"bilinear"`` can add spurious high-frequency banding.
    show : bool
        If True, display the animation interactively.
    """
    snapshots = discover_snapshots(snap_dir, comp)
    if not snapshots:
        raise FileNotFoundError(
            f"No snapshot files matching '*{comp}.su' found in {snap_dir}"
        )

    if nframes is not None:
        snapshots = snapshots[:nframes]

    print(f"Found {len(snapshots)} {comp} snapshot files in {snap_dir}")

    # Load grid metadata if available
    grid = None
    if fd_model_path is not None and fd_model_path.exists():
        grid = parse_fd_model(fd_model_path)
        print(f"  Grid: x=[{grid['x0']:.0f}, {grid['xn']:.0f}] m, "
              f"z=[{grid['z0']:.0f}, {grid['zn']:.0f}] m, "
              f"dx={grid['dx']:.0f} m, dz={grid['dz']:.0f} m")

    def _frame_for_plot(raw: np.ndarray) -> np.ndarray:
        """Orient to (nz, nx) and optionally drop surface ghost rows."""
        field = raw.T
        return snapshot_for_plot(field, iz_plot_start)

    # Load first frame to determine dimensions and color scale
    first_data = _frame_for_plot(load_snapshot(snapshots[0][1]))
    nx = first_data.shape[1]
    nz = first_data.shape[0]
    print(f"  Snapshot shape (plotted): nz={nz}, nx={nx}"
          + (f" (dropped {iz_plot_start} surface row(s))" if iz_plot_start else ""))

    if vmin is None or vmax is None:
        # Sample a few frames for robust color scaling
        sample_indices = np.linspace(0, len(snapshots) - 1,
                                     min(10, len(snapshots)), dtype=int)
        peak = 0.0
        for idx in sample_indices:
            d = _frame_for_plot(load_snapshot(snapshots[idx][1]))
            peak = max(peak, np.max(np.abs(d)))
        if peak < 1e-30:
            peak = 1.0
        if vmin is None:
            vmin = -0.8 * peak
        if vmax is None:
            vmax = 0.8 * peak

    # Set up axis extents
    if grid is not None:
        z_top_km = (grid["z0"] + iz_plot_start * grid["dz"]) / 1000.0
        extent = [grid["x0"] / 1000, grid["xn"] / 1000,
                  grid["zn"] / 1000, z_top_km]
        xlabel = "X (km)"
        ylabel = "Depth (km)"
    else:
        extent = [0, nx, nz + iz_plot_start, iz_plot_start]
        xlabel = "X (grid)"
        ylabel = "Z (grid)"

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        first_data, aspect="auto", cmap=cmap,
        vmin=vmin, vmax=vmax, extent=extent, interpolation=interpolation,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"Displacement {comp}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if grid is not None:
        dt_frame = grid["dt"] * grid["nstep"]
    else:
        dt_frame = 1.0

    time_text = ax.set_title(
        f"{comp} wavefield, t = {snapshots[0][0] * (grid['dt'] if grid else 1.0):.3f} s"
    )

    def update(frame_idx):
        it, path = snapshots[frame_idx]
        data = _frame_for_plot(load_snapshot(path))
        im.set_data(data)
        t_sec = it * (grid["dt"] if grid else 1.0)
        ax.set_title(f"{comp} wavefield, t = {t_sec:.3f} s")
        return [im]

    anim = FuncAnimation(
        fig, update, frames=len(snapshots),
        interval=1000 // fps, blit=False,
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()
        if suffix == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(str(out_path), writer=writer, dpi=dpi)
        print(f"Saved animation to {out_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate FDFK2D wavefield snapshots."
    )
    parser.add_argument("snap_dir", type=Path,
                        help="Directory containing snapshot SU files.")
    parser.add_argument("--comp", choices=["ux", "uz"], default="uz",
                        help="Component to animate (default: uz).")
    parser.add_argument("--fd-model", type=Path, default=None,
                        help="Path to FD_model.dat for axis metadata.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output file (.mp4 or .gif).")
    parser.add_argument("--fps", type=int, default=20,
                        help="Frames per second (default: 20).")
    parser.add_argument("--dpi", type=int, default=100,
                        help="Output resolution (default: 100).")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Color scale min (auto if omitted).")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Color scale max (auto if omitted).")
    parser.add_argument("--cmap", default="seismic",
                        help="Colormap (default: seismic).")
    parser.add_argument("--nframes", type=int, default=None,
                        help="Limit to first N frames.")
    parser.add_argument("--show", action="store_true",
                        help="Display animation interactively.")
    args = parser.parse_args()

    # Auto-detect FD_model.dat if not provided
    fd_model = args.fd_model
    if fd_model is None:
        candidates = [
            args.snap_dir.parent / "input" / "FD_model.dat",
            args.snap_dir.parent / "FD_model.dat",
            args.snap_dir / ".." / "input" / "FD_model.dat",
        ]
        for c in candidates:
            if c.exists():
                fd_model = c
                print(f"Auto-detected FD_model.dat: {fd_model}")
                break

    animate_wavefield(
        args.snap_dir,
        comp=args.comp,
        fd_model_path=fd_model,
        out_path=args.out,
        fps=args.fps,
        dpi=args.dpi,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        nframes=args.nframes,
        show=args.show,
    )


if __name__ == "__main__":
    main()
