#!/usr/bin/env python3
"""Shared hybrid wavefield + receiver-function workflow for example cases."""

from __future__ import annotations

import glob
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from fk.engine import compute_fk_seismograms
from fk.geometry import rayp_deg_to_si
from fk.io_fortran import load_example, read_fk_model, read_su, write_su
from fk.model import FKLayerModel
from hybrid.config import HybridConfig
from hybrid.engine import compute_hybrid_seismograms
from plots.animate_wavefield import animate_wavefield
from plots.rf import rf_amp_percentile, rf_power_law, sac_power_wiggle
from plots.wavefield_snapshots import (
    annotate_interfaces,
    model_interface_profiles,
    snapshot_for_plot,
)
from rf import compute_prf, compute_srf


@dataclass(frozen=True)
class ExampleConfig:
    """Per-example settings for the RF workflow."""

    case_name: str
    title_label: str
    vs_depth_max_km: float
    force_regen: bool = False
    force_snapshot_regen: bool = False
    force_rf_regen: bool = False
    bwf_pad_factor: float = 1.0


SNAPSHOT_MAX_COLUMNS = 5
SNAPSHOT_PERCENTILE = 99.5
SNAPSHOT_IZ_PLOT_START = 1
RF_F0 = 8.0
RF_ITMAX = 100
RF_MINDERR = 0.001
RF_TSHIFT_P = 10.0
RF_TSHIFT_S = 6.5
T_RF_MIN = -1.0
T_RF_MAX = 8.0
RF_POWER = 2.0
RF_AMP_PCT = 80.0
BIN_SIZE_KM = 0.08
DECIMATE_RF = 1
N_RCV_PLOT = 5
GIF_FPS = 15
GIF_DPI = 100


def _model_stems(inpar_path: Path) -> tuple[str, str]:
    lines = [ln.strip() for ln in inpar_path.read_text().splitlines() if ln.strip()]
    return lines[5], lines[6]


def _without_pml_path(input_dir: Path, stem: str) -> Path:
    if stem.endswith(".su"):
        stem = stem[:-3]
    path = input_dir / f"{stem}_without_pml.su"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; rebuild the model.")
    return path


def choose_snapshot_indices(n_snaps: int, max_cols: int = SNAPSHOT_MAX_COLUMNS) -> list[int]:
    if n_snaps <= max_cols:
        return list(range(n_snaps))
    idx = np.linspace(0, n_snaps - 1, max_cols, dtype=int)
    return [int(i) for i in idx]


def symmetric_limit(arrays: list[np.ndarray], percentile: float = SNAPSHOT_PERCENTILE) -> float:
    values = np.concatenate([np.ravel(np.abs(a[np.isfinite(a)])) for a in arrays])
    if values.size == 0:
        return 1.0
    limit = float(np.percentile(values, percentile))
    return limit if limit > 1e-30 else 1.0


def overlay_trace(tr: np.ndarray) -> np.ndarray:
    return tr / (np.max(np.abs(tr)) + 1e-30)


def _halfspace_velocity(model: FKLayerModel, incidence: str) -> float:
    """Reference half-space velocity for plane-wave init depth."""
    half = model.halfspace_index
    if incidence.upper() == "P":
        return float(model.vp[half])
    return float(model.vs[half])


def _init_depth_z(
    p_si: float,
    v_half: float,
    profile_length: float,
    model_max_depth: float,
) -> float:
    sin_arg = min(1.0, max(-1.0, p_si * v_half))
    return model_max_depth + 0.5 * profile_length * np.tan(np.arcsin(sin_arg))


def _rf_decon(incidence: str):
    """Return (decon_fn, tshift, phase_label) for P or S incidence."""
    if incidence.upper() == "S":
        return compute_srf, RF_TSHIFT_S, "S-wave"
    return compute_prf, RF_TSHIFT_P, "P-wave"


def _plot_velocity_section(
    velocity: np.ndarray,
    *,
    label: str,
    cbar_label: str,
    extent_km: list[float],
    interface_profiles,
    out_png: Path,
    title: str,
) -> None:
    with plt.rc_context({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=(10, 3.6))
        im = ax.imshow(
            velocity,
            aspect="auto",
            cmap="viridis",
            extent=extent_km,
            interpolation="nearest",
            rasterized=True,
        )
        annotate_interfaces(
            ax, interface_profiles, color="white", lw=0.7, alpha=0.75,
        )
        ax.set_xlim(extent_km[0], extent_km[1])
        ax.set_ylim(extent_km[2], extent_km[3])
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Depth (km)")
        ax.set_title(title, pad=8)
        ax.tick_params(labelsize=9, length=3, color="0.25")
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("0.35")
        cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.035)
        cb.set_label(cbar_label, fontsize=10)
        cb.ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out_png}")


def _shift_traces_time(
    traces: np.ndarray, dt: float, shifts_s: np.ndarray,
) -> np.ndarray:
    """Shift each trace by ``shifts_s[i]`` seconds (positive = delay)."""
    nrcv, nt = traces.shape
    out = np.empty_like(traces)
    freq = np.fft.rfftfreq(nt, dt)
    for i in range(nrcv):
        spec = np.fft.rfft(traces[i])
        spec *= np.exp(-2j * np.pi * freq * shifts_s[i])
        out[i] = np.fft.irfft(spec, n=nt)
    return out


def _align_plane_wave_moveout(
    seisx: np.ndarray,
    seisz: np.ndarray,
    dt: float,
    rx: np.ndarray,
    x0_plane: float,
    p_si: float,
    phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-time hybrid traces to per-receiver plane-wave reference before RF decon.

    Hybrid synthesis uses profile-center ``x0_plane``; undo that moveout so each
    station's direct arrival aligns like ``x0_plane=rx[i]`` in the FK engine.
    """
    p_along = -p_si * np.cos(phi)
    shifts = -p_along * (rx - x0_plane)
    return (
        _shift_traces_time(seisx, dt, shifts),
        _shift_traces_time(seisz, dt, shifts),
    )


def _cache_case_marker(cache_dir: Path) -> Path:
    return cache_dir / ".case"


def _cache_matches_case(cache_dir: Path, case_name: str) -> bool:
    """Return True if cached simulation outputs belong to ``case_name``."""
    marker = _cache_case_marker(cache_dir)
    if not marker.exists():
        return False
    return marker.read_text().strip() == case_name


def _write_cache_case_marker(cache_dir: Path, case_name: str) -> None:
    _cache_case_marker(cache_dir).write_text(f"{case_name}\n")


def _invalidate_stale_simulation_cache(cache_dir: Path, case_name: str) -> None:
    """Drop seismogram/snapshot/RF cache copied from another example case."""
    if _cache_matches_case(cache_dir, case_name):
        return
    for name in ("seisx.su", "seisz.su"):
        (cache_dir / name).unlink(missing_ok=True)
    snap_cache = cache_dir / "snapshots"
    if snap_cache.exists():
        shutil.rmtree(snap_cache)


def run_example_workflow(example_dir: Path, cfg: ExampleConfig) -> None:
    """Run the full Antarctica_RF-style workflow for one example folder."""
    example_dir = example_dir.resolve()
    project_root = example_dir.parent.parent
    sys.path.insert(0, str(project_root))

    out_dir = example_dir
    cache_dir = out_dir / "cache"
    model_dir = cache_dir / cfg.case_name / "input"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model input not found: {model_dir}")

    _invalidate_stale_simulation_cache(cache_dir, cfg.case_name)
    cache_valid = _cache_matches_case(cache_dir, cfg.case_name)

    print(f"Project root : {project_root}")
    print(f"Example dir  : {example_dir}")
    print(f"Model input  : {model_dir}")

    ex = load_example(model_dir, side="left")
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

    vp_stem, vs_stem = _model_stems(model_dir / "inpar.dat")
    vp_tr, _ = read_su(_without_pml_path(model_dir, vp_stem))
    vs_tr, _ = read_su(_without_pml_path(model_dir, vs_stem))
    vp_2d = vp_tr.T.astype(np.float64)
    vs_2d = vs_tr.T.astype(np.float64)
    rho_2d = (vp_2d + 980.0) / 2.760

    model_left = read_fk_model(model_dir / "FK_model_left.dat")
    model_right = read_fk_model(model_dir / "FK_model_right.dat")
    p_si = rayp_deg_to_si(ex.src.rayp_deg)

    v_half = _halfspace_velocity(model_left, ex.source.incidence)
    z_init = _init_depth_z(p_si, v_half, (nx - 1) * fd.dx, fd.zn)
    rf_decon, rf_tshift, rf_phase_label = _rf_decon(ex.source.incidence)
    snap_interval = fd.nstep

    print(
        f"Grid: nx={nx}, nz={nz}, nt={fd.nt}, "
        f"source={ex.source.incidence}-wave f0={ex.source.f0} Hz"
    )

    extent_km = [config.x0 / 1000, config.xn / 1000, config.zn / 1000, config.z0 / 1000]
    extent_vs_km = [extent_km[0], extent_km[1], cfg.vs_depth_max_km, extent_km[3]]
    iz_max = min(vs_2d.shape[0], int(round(cfg.vs_depth_max_km * 1000 / fd.dz)) + 1)
    vs_shallow = vs_2d[:iz_max, :]
    vp_shallow = vp_2d[:iz_max, :]
    interface_profiles_shallow = model_interface_profiles(
        vs_2d, config.x0, config.dx, fd.z0, fd.dz, max_depth_km=cfg.vs_depth_max_km,
    )
    depth_tag = f"0–{cfg.vs_depth_max_km:g} km"

    vs_png = out_dir / "vs_model_section.png"
    _plot_velocity_section(
        vs_shallow,
        label="Vs",
        cbar_label="Vs (m/s)",
        extent_km=extent_vs_km,
        interface_profiles=interface_profiles_shallow,
        out_png=vs_png,
        title=f"{cfg.title_label} S-wave velocity model ({depth_tag})",
    )
    vp_png = out_dir / "vp_model_section.png"
    _plot_velocity_section(
        vp_shallow,
        label="Vp",
        cbar_label="Vp (m/s)",
        extent_km=extent_vs_km,
        interface_profiles=interface_profiles_shallow,
        out_png=vp_png,
        title=f"{cfg.title_label} P-wave velocity model ({depth_tag})",
    )
    with plt.rc_context({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(1, 2, figsize=(14, 3.8), sharey=True)
        for ax, data, cbar_label, subtitle in zip(
            axes,
            (vp_shallow, vs_shallow),
            ("Vp (m/s)", "Vs (m/s)"),
            ("P-wave velocity", "S-wave velocity"),
            strict=True,
        ):
            im = ax.imshow(
                data,
                aspect="auto",
                cmap="viridis",
                extent=extent_vs_km,
                interpolation="nearest",
                rasterized=True,
            )
            annotate_interfaces(
                ax, interface_profiles_shallow, color="white", lw=0.7, alpha=0.75,
            )
            ax.set_xlim(extent_vs_km[0], extent_vs_km[1])
            ax.set_ylim(extent_vs_km[2], extent_vs_km[3])
            ax.set_xlabel("Distance (km)")
            ax.set_title(f"{subtitle} ({depth_tag})", pad=8)
            ax.tick_params(labelsize=9, length=3, color="0.25")
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("0.35")
            cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.04)
            cb.set_label(cbar_label, fontsize=10)
            cb.ax.tick_params(labelsize=8)
        axes[0].set_ylabel("Depth (km)")
        fig.suptitle(f"{cfg.title_label} velocity model sections", fontsize=13, y=1.02)
        fig.tight_layout()
        vel_png = out_dir / "velocity_model_sections.png"
        fig.savefig(vel_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {vel_png}")

    seisx_path = cache_dir / "seisx.su"
    seisz_path = cache_dir / "seisz.su"
    snap_cache = cache_dir / "snapshots"
    snap_marker = snap_cache / ".done"

    need_seis = (
        cfg.force_regen
        or not cache_valid
        or not (seisx_path.exists() and seisz_path.exists())
    )
    need_snaps = (
        cfg.force_snapshot_regen
        or cfg.force_regen
        or not cache_valid
        or not snap_marker.exists()
    )

    if need_seis or need_snaps:
        snap_iv = snap_interval if need_snaps else 0
        if snap_iv > 0:
            print(
                f"Running compute_hybrid_seismograms "
                f"(seismograms + snapshots, Numba fused loop, interval={snap_iv})..."
            )
        else:
            print("Running compute_hybrid_seismograms (Numba fused loop)...")
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
            bwf_pad_factor=cfg.bwf_pad_factor,
            verbose=True,
            snapshot_interval=snap_iv,
        )
        print(f"Wall time: {time.perf_counter() - t0:.1f} s")

        if need_seis or snap_iv > 0:
            write_su(seisx_path, result.ux, config.dt)
            write_su(seisz_path, result.uz, config.dt)
            _write_cache_case_marker(cache_dir, cfg.case_name)
            cache_valid = True
        seisx, seisz = result.ux, result.uz
        t = result.t

        if need_snaps and snap_iv > 0:
            snap_cache.mkdir(parents=True, exist_ok=True)
            for i, (ux_snap, uz_snap) in enumerate(
                zip(result.snapshots_ux, result.snapshots_uz)
            ):
                it = i * snap_interval
                write_su(
                    snap_cache / f"{it:05d}ux.su",
                    ux_snap.T.astype(np.float32),
                    config.dt,
                )
                write_su(
                    snap_cache / f"{it:05d}uz.su",
                    uz_snap.T.astype(np.float32),
                    config.dt,
                )
            snap_marker.write_text("ok")
            snapshots_ux = result.snapshots_ux
            snapshots_uz = result.snapshots_uz
        elif snap_marker.exists():
            print(f"Loading cached snapshots from {snap_cache}")
            ux_files = sorted(glob.glob(str(snap_cache / "*ux.su")))
            snapshots_ux, snapshots_uz = [], []
            for f in ux_files:
                stem = Path(f).name.replace("ux.su", "")
                ux_tr, _ = read_su(f)
                uz_tr, _ = read_su(snap_cache / f"{stem}uz.su")
                snapshots_ux.append(ux_tr.T)
                snapshots_uz.append(uz_tr.T)
        else:
            snapshots_ux, snapshots_uz = [], []
    else:
        print(f"Loading cached seismograms from {cache_dir}")
        seisx, dt_x = read_su(seisx_path)
        seisz, dt_z = read_su(seisz_path)
        assert abs(dt_x - config.dt) < 1e-9
        t = np.arange(seisx.shape[1]) * config.dt

        print(f"Loading cached snapshots from {snap_cache}")
        ux_files = sorted(glob.glob(str(snap_cache / "*ux.su")))
        snapshots_ux, snapshots_uz = [], []
        for f in ux_files:
            stem = Path(f).name.replace("ux.su", "")
            ux_tr, _ = read_su(f)
            uz_tr, _ = read_su(snap_cache / f"{stem}uz.su")
            snapshots_ux.append(ux_tr.T)
            snapshots_uz.append(uz_tr.T)

    nrcv, _ = seisx.shape
    print(f"Snapshots available: {len(snapshots_uz)}")

    extent_plot_km = [
        extent_km[0], extent_km[1], extent_km[2],
        (config.z0 + config.dz) / 1000,
    ]

    def style_snapshot_axis(ax, *, show_xlabel: bool, show_ylabel: bool) -> None:
        ax.set_xlim(extent_plot_km[0], extent_plot_km[1])
        ax.set_ylim(extent_plot_km[2], extent_plot_km[3])
        ax.tick_params(labelsize=9, length=3, color="0.25")
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("0.35")
        if show_xlabel:
            ax.set_xlabel("Distance (km)", fontsize=10)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if show_ylabel:
            ax.set_ylabel("Depth (km)", fontsize=10)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    idxs = choose_snapshot_indices(len(snapshots_uz))
    ncols = len(idxs)
    wave_limit = symmetric_limit(
        [snapshot_for_plot(snapshots_uz[i]) for i in idxs]
        + [snapshot_for_plot(snapshots_ux[i]) for i in idxs],
        SNAPSHOT_PERCENTILE,
    )
    interface_profiles = model_interface_profiles(
        vs_2d, config.x0, config.dx, fd.z0, fd.dz,
    )
    wave_norm = Normalize(vmin=-wave_limit, vmax=wave_limit)
    fig_width = max(11.0, 2.65 * ncols + 1.4)
    fig_height = 2.15 * 2 + 1.1

    with plt.rc_context({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(
            2, ncols, figsize=(fig_width, fig_height),
            sharex=True, sharey=True, squeeze=False, constrained_layout=False,
        )
        fig.subplots_adjust(left=0.08, right=0.86, top=0.84, bottom=0.12,
                          wspace=0.04, hspace=0.08)
        wave_im = None
        for col, si in enumerate(idxs):
            t_sec = si * snap_interval * config.dt
            axes[0, col].set_title(f"t = {t_sec:.1f} s", pad=8)
            for row, comp_snapshots in enumerate((snapshots_uz, snapshots_ux)):
                data = snapshot_for_plot(comp_snapshots[si])
                im = axes[row, col].imshow(
                    data, aspect="auto", cmap="RdBu_r", norm=wave_norm,
                    extent=extent_plot_km, interpolation="nearest", rasterized=True,
                )
                if wave_im is None:
                    wave_im = im
                annotate_interfaces(axes[row, col], interface_profiles)
                style_snapshot_axis(
                    axes[row, col],
                    show_xlabel=(row == 1),
                    show_ylabel=(col == 0),
                )
                if col == 0:
                    axes[row, col].text(
                        -0.16, 0.5, ("uz", "ux")[row],
                        transform=axes[row, col].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=11, fontweight="semibold",
                    )
        fig.suptitle(
            f"{cfg.title_label} wavefield snapshots", fontsize=14, fontweight="semibold", y=1.0,
        )
        if wave_im is not None:
            top_pos = axes[0, -1].get_position()
            bot_pos = axes[1, -1].get_position()
            cax_wave = fig.add_axes([
                top_pos.x1 + 0.012, bot_pos.y0, 0.015, top_pos.y1 - bot_pos.y0,
            ])
            cb = fig.colorbar(wave_im, cax=cax_wave)
            cb.set_label("displacement (m)", fontsize=10)
            cb.ax.tick_params(labelsize=8)
        snap_png = out_dir / "wavefield_snapshots.png"
        fig.savefig(snap_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {snap_png}")

    gif_limit = symmetric_limit(
        [snapshot_for_plot(s) for s in snapshots_uz],
        SNAPSHOT_PERCENTILE,
    )
    gif_path = out_dir / "wavefield_uz.gif"
    animate_wavefield(
        snap_cache,
        comp="uz",
        fd_model_path=model_dir / "FD_model.dat",
        out_path=gif_path,
        fps=GIF_FPS,
        dpi=GIF_DPI,
        vmin=-gif_limit,
        vmax=gif_limit,
        cmap="RdBu_r",
        iz_plot_start=SNAPSHOT_IZ_PLOT_START,
        interpolation="nearest",
    )
    print(f"Saved {gif_path}")

    rcv_plot_idx = np.linspace(0, nrcv - 1, N_RCV_PLOT, dtype=int)
    fig, axes = plt.subplots(len(rcv_plot_idx), 2, figsize=(14, 3 * len(rcv_plot_idx)), sharex=True)
    for row, idx in enumerate(rcv_plot_idx):
        x_km = ex.rx[idx] / 1000
        axes[row, 0].plot(t, overlay_trace(seisz[idx]), "k-", lw=0.8)
        axes[row, 0].set_ylabel(f"x = {x_km:.1f} km")
        if row == 0:
            axes[row, 0].set_title("uz (vertical)")
        axes[row, 1].plot(t, overlay_trace(seisx[idx]), "k-", lw=0.8)
        if row == 0:
            axes[row, 1].set_title("ux (horizontal)")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(f"{cfg.title_label} receiver waveforms (peak-normalized)", y=1.01)
    fig.tight_layout()
    rcv_png = out_dir / "receiver_waveforms.png"
    fig.savefig(rcv_png, dpi=150)
    plt.close(fig)
    print(f"Saved {rcv_png}")

    x_km = ex.rx / 1000.0

    def plot_moveout(data: np.ndarray, comp: str, out_png: Path, decimate: int = 2, scale: float = 0.6) -> None:
        indices = np.arange(0, nrcv, decimate)
        dx_km = (x_km[1] - x_km[0]) if nrcv > 1 else 0.1
        fig, ax = plt.subplots(figsize=(10, 7))
        for idx in indices:
            tr = overlay_trace(data[idx])
            y0 = x_km[idx]
            y = y0 + tr * dx_km * scale
            ax.plot(t, y, "k", lw=0.5)
            ax.fill_between(t, y0, y, where=(y > y0), color="k", alpha=0.5, linewidth=0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance along profile (km)")
        ax.set_title(f"{cfg.title_label} {comp} moveout (peak-normalized wiggles)")
        ax.set_ylim(x_km[0] - dx_km, x_km[-1] + dx_km)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved {out_png}")

    plot_moveout(seisz, "uz", out_dir / "moveout_uz.png")
    plot_moveout(seisx, "ux", out_dir / "moveout_ux.png")

    x0_plane_hybrid = 0.5 * (config.x0 + config.xn)
    seisx_rf, seisz_rf = _align_plane_wave_moveout(
        seisx, seisz, config.dt, ex.rx, x0_plane_hybrid, p_si, ex.phi,
    )

    rf_npz = out_dir / "receiver_functions.npz"
    rf_cache_ok = False
    if rf_npz.exists() and not (cfg.force_regen or cfg.force_rf_regen):
        cached = np.load(rf_npz)
        rf_cache_ok = (
            cache_valid
            and "incidence" in cached
            and str(cached["incidence"]) == ex.source.incidence
        )
    if (
        cfg.force_regen
        or cfg.force_rf_regen
        or not rf_cache_ok
        or not rf_npz.exists()
    ):
        print(f"Computing {rf_phase_label} receiver functions (moveout-aligned)...")
        rfs, t_rf, rms_arr = rf_decon(
            seisx_rf, seisz_rf, config.dt,
            tshift=rf_tshift, f0=RF_F0,
            method="iter", itmax=RF_ITMAX, minderr=RF_MINDERR,
        )
        np.savez(
            rf_npz, rfs=rfs, t_rf=t_rf, dt=config.dt, rms=rms_arr,
            f0=RF_F0, tshift=rf_tshift, incidence=ex.source.incidence,
        )
        print(f"Saved RF data to {rf_npz}")
    else:
        cached = np.load(rf_npz)
        rfs, t_rf, rms_arr = cached["rfs"], cached["t_rf"], cached["rms"]
        print(f"Loaded cached RF from {rf_npz}")

    disp_mask = (t_rf >= T_RF_MIN) & (t_rf <= T_RF_MAX)
    amp_mask = (t_rf >= 0.0) & (t_rf <= T_RF_MAX)
    rfs_pow = rf_power_law(rfs, RF_POWER)
    amp_ref = rf_amp_percentile(rfs_pow, mask=amp_mask, percentile=RF_AMP_PCT)
    t_plot = t_rf[disp_mask]
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx in range(0, nrcv, DECIMATE_RF):
        wiggle = sac_power_wiggle(rfs[idx], amp_ref, BIN_SIZE_KM, power=RF_POWER)[disp_mask]
        x_pos = x_km[idx]
        ax.plot(x_pos + wiggle, t_plot, "k", lw=0.4)
        ax.fill_betweenx(t_plot, x_pos, x_pos + wiggle, where=(wiggle > 0),
                         color="red", alpha=0.45, linewidth=0)
        ax.fill_betweenx(t_plot, x_pos, x_pos + wiggle, where=(wiggle < 0),
                         color="blue", alpha=0.45, linewidth=0)
    ax.axhline(0.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Distance along profile (km)")
    ax.set_ylabel("Time lag (s)")
    ax.set_title(
        f"{rf_phase_label} RF section — {cfg.title_label} (f0={RF_F0}, iter, n={nrcv})\n"
        "SAC style: red = positive, blue = negative"
    )
    ax.set_ylim(T_RF_MAX, T_RF_MIN)
    ax.set_xlim(x_km[0] - BIN_SIZE_KM, x_km[-1] + BIN_SIZE_KM)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    rf_png = out_dir / "rf_section_sac.png"
    fig.savefig(rf_png, dpi=150)
    plt.close(fig)
    print(f"Saved {rf_png}")

    def column_to_fk_model(ix: int, z0: float, dz: float) -> FKLayerModel:
        vp_col = vp_2d[:, ix]
        vs_col = vs_2d[:, ix]
        rho_col = rho_2d[:, ix]
        nz_col = vp_col.size
        z_depths = z0 + np.arange(nz_col) * dz
        changed = np.ones(nz_col, dtype=bool)
        changed[1:] = (np.abs(np.diff(vp_col)) > 1e-3) | (np.abs(np.diff(vs_col)) > 1e-3)
        layer_idx = np.where(changed)[0]
        return FKLayerModel.from_layers(
            vp_col[layer_idx], vs_col[layer_idx], z_depths[layer_idx], rho=rho_col[layer_idx],
        )

    def receiver_column_index(x_m: float) -> int:
        ix = int(round((x_m - config.x0) / config.dx))
        return int(np.clip(ix, 0, vp_2d.shape[1] - 1))

    rf_id_npz = out_dir / "receiver_functions_1d_id.npz"
    rf_id_cache_ok = False
    if rf_id_npz.exists() and not (cfg.force_regen or cfg.force_rf_regen):
        cached_id_probe = np.load(rf_id_npz)
        rf_id_cache_ok = (
            cache_valid
            and "incidence" in cached_id_probe
            and str(cached_id_probe["incidence"]) == ex.source.incidence
        )
    if (
        cfg.force_regen
        or cfg.force_rf_regen
        or not rf_id_cache_ok
        or not rf_id_npz.exists()
    ):
        print("Computing 1D-ID FK synthetics per receiver...")
        seisx_id = np.zeros((nrcv, config.nt), dtype=np.float64)
        seisz_id = np.zeros((nrcv, config.nt), dtype=np.float64)
        t0_id = time.perf_counter()
        for i in range(nrcv):
            ix = receiver_column_index(ex.rx[i])
            model_id = column_to_fk_model(ix, fd.z0, fd.dz)
            # Per-receiver x0_plane: re-time the plane wave so each station sees
            # the direct arrival at the same relative origin. Global x0_plane
            # lets oblique moveout push energy past tmax (~160 km on test1) and
            # corrupts iterative RF deconvolution.
            fk_res = compute_fk_seismograms(
                model_id, ex.source, [ex.rx[i]], config.dt, config.nt,
                p_si=p_si, receivers_z=[ex.rz[i]], x0_plane=ex.rx[i],
                phi=ex.phi, model_max_depth=fd.zn, z_init=z_init, method="kennett_numba",
            )
            seisx_id[i] = fk_res.ux[0]
            seisz_id[i] = fk_res.uz[0]
        print(f"FK synthetics wall time: {time.perf_counter() - t0_id:.1f} s")
        rfs_id, t_rf_id, rms_id = rf_decon(
            seisx_id, seisz_id, config.dt,
            tshift=rf_tshift, f0=RF_F0, method="iter", itmax=RF_ITMAX, minderr=RF_MINDERR,
        )
        np.savez(
            rf_id_npz, rfs=rfs_id, t_rf=t_rf_id, seisx=seisx_id, seisz=seisz_id,
            rms=rms_id, dt=config.dt, incidence=ex.source.incidence,
        )
        print(f"Saved 1D-ID RF data to {rf_id_npz}")
    else:
        cached_id = np.load(rf_id_npz)
        rfs_id, t_rf_id = cached_id["rfs"], cached_id["t_rf"]
        print(f"Loaded cached 1D-ID RF from {rf_id_npz}")

    disp_mask_id = (t_rf_id >= T_RF_MIN) & (t_rf_id <= T_RF_MAX)
    amp_mask_id = (t_rf_id >= 0.0) & (t_rf_id <= T_RF_MAX)
    rfs_id_pow = rf_power_law(rfs_id, RF_POWER)
    amp_ref_id = rf_amp_percentile(rfs_id_pow, mask=amp_mask_id, percentile=RF_AMP_PCT)
    t_plot_id = t_rf_id[disp_mask_id]
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx in range(nrcv):
        wiggle = sac_power_wiggle(rfs_id[idx], amp_ref_id, BIN_SIZE_KM, power=RF_POWER)[disp_mask_id]
        x_pos = x_km[idx]
        ax.plot(x_pos + wiggle, t_plot_id, "k", lw=0.4)
        ax.fill_betweenx(t_plot_id, x_pos, x_pos + wiggle, where=(wiggle > 0),
                         color="red", alpha=0.45, linewidth=0)
        ax.fill_betweenx(t_plot_id, x_pos, x_pos + wiggle, where=(wiggle < 0),
                         color="blue", alpha=0.45, linewidth=0)
    ax.axhline(0.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Distance along profile (km)")
    ax.set_ylabel("Time lag (s)")
    ax.set_title(
        f"1D-ID {rf_phase_label} RF section — {cfg.title_label} (f0={RF_F0}, n={nrcv})\n"
        "SAC style: red = positive, blue = negative"
    )
    ax.set_ylim(T_RF_MAX, T_RF_MIN)
    ax.set_xlim(x_km[0] - BIN_SIZE_KM, x_km[-1] + BIN_SIZE_KM)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    rf_id_png = out_dir / "rf_section_1d_id.png"
    fig.savefig(rf_id_png, dpi=150)
    plt.close(fig)
    print(f"Saved {rf_id_png}")

    mid = nrcv // 2
    rf_mid = rfs[mid][disp_mask]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_plot, rf_mid, "k", lw=1.0)
    ax.fill_between(t_plot, rf_mid, where=(rf_mid > 0), color="red", alpha=0.4)
    ax.fill_between(t_plot, rf_mid, where=(rf_mid < 0), color="blue", alpha=0.4)
    ax.axvline(0.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{rf_phase_label} RF — trace {mid} (x = {ex.rx[mid]/1000:.1f} km)")
    ax.set_xlim(T_RF_MIN, T_RF_MAX)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    single_png = out_dir / "rf_single_mid.png"
    fig.savefig(single_png, dpi=150)
    plt.close(fig)
    print(f"Saved {single_png}")
    print(f"\nAll figures written to: {out_dir}")
