#!/usr/bin/env python3
"""Run benchmark logic and save PyRaysum RFs (no Jupyter required).

Computes RFs two ways for reference:
- **Spectral division** — original PyRaysum ``rf=True`` / telewavesim FFT workflow
- **seispy iter** — ``compute_prf`` / ``compute_srf`` (f0=8.0), FK-compatible (saved for cross-benchmark)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift

NOTEBOOK_DIR = Path(__file__).resolve().parent
ROOT = NOTEBOOK_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from fk.geometry import rayp_deg_to_si
from rf import compute_prf, compute_srf

import pyraysum as prs
from telewavesim import utils as tws_utils

DT = 0.025
NT = 2048
P_DEG = 4.798
S_DEG = 9.535
TSHIFT_P = 5.0
TSHIFT_S = 6.5
F0_RF = 8.0
ITMAX = 400
MINDERR = 0.001
T_MAX_P = 30.0
T_MAX_S = 5.0
MULTS = 2  # all first-order free-surface multiples (PyRaysum Control.mults)
OUT_DIR = NOTEBOOK_DIR / "output/pyraysum_rfs"
FIG_DIR = NOTEBOOK_DIR / "output/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MANTLE = {"vp": 8.0, "vs": 4.45}
LAB_LID = {"vp": 8.1, "vs": 4.5}
LAB_LVZ = {"vp": 7.6, "vs": 4.2}

MODELS = {
    "M1": {"title": "M1: Single-layer crust (35 km)", "vp": [6.0, MANTLE["vp"]], "vs": [3.45, MANTLE["vs"]], "z_top": [0.0, 35.0]},
    "M2": {"title": "M2: Two-layer crust (20 + 15 km)", "vp": [6.0, 6.5, MANTLE["vp"]], "vs": [3.45, 3.75, MANTLE["vs"]], "z_top": [0.0, 20.0, 35.0]},
    "M3": {"title": "M3: Three-layer crust (5 + 15 + 15 km)", "vp": [4.0, 6.0, 6.8, MANTLE["vp"]], "vs": [2.3, 3.45, 3.9, MANTLE["vs"]], "z_top": [0.0, 5.0, 20.0, 35.0]},
    "M4": {"title": "M4: 2 km ice + single crust", "vp": [4.0, 6.0, MANTLE["vp"]], "vs": [2.0, 3.45, MANTLE["vs"]], "z_top": [0.0, 2.0, 37.0]},
    "M5": {"title": "M5: 2 km ice + 0.5 km sediment + single crust", "vp": [4.0, 2.8, 6.0, MANTLE["vp"]], "vs": [2.0, 1.6, 3.45, MANTLE["vs"]], "z_top": [0.0, 2.0, 2.5, 37.5]},
    "M6": {"title": "M6: Single crust + LAB", "vp": [6.0, LAB_LID["vp"], LAB_LVZ["vp"]], "vs": [3.45, LAB_LID["vs"], LAB_LVZ["vs"]], "z_top": [0.0, 35.0, 100.0]},
    "M7": {"title": "M7: Two-layer crust + LAB", "vp": [6.0, 6.5, LAB_LID["vp"], LAB_LVZ["vp"]], "vs": [3.45, 3.75, LAB_LID["vs"], LAB_LVZ["vs"]], "z_top": [0.0, 20.0, 35.0, 100.0]},
}


def _rho_list(vp_km_s):
    vp_m = np.array(vp_km_s, dtype=float) * 1000.0
    return ((vp_m + 980.0) / 2.760).tolist()


def _layer_thickness_km(z_top_km):
    z = np.array(z_top_km, dtype=float)
    thick = np.diff(np.append(z, np.inf))
    thick[-1] = 0.0
    return thick.tolist()


def build_pyraysum_model(spec):
    rho = _rho_list(spec["vp"])
    thick_m = [t * 1000.0 for t in _layer_thickness_km(spec["z_top"])]
    vp_m = [v * 1000.0 for v in spec["vp"]]
    vs_m = [v * 1000.0 for v in spec["vs"]]
    return prs.Model(thick_m, rho, vp_m, vs_m, flag=1), rho, thick_m


def build_tws_model(spec):
    rho = _rho_list(spec["vp"])
    thick_km = _layer_thickness_km(spec["z_top"])
    n = len(thick_km)
    z = [0.0] * n
    return tws_utils.Model(thick_km, rho, spec["vp"], spec["vs"], ["iso"] * n, z, z, z)


def _rtz_from_pyraysum_seis(seis):
    by_comp = {tr.stats.channel[-1]: tr.data.astype(np.float64) for tr in seis}
    return by_comp["R"], by_comp["Z"]


def _pyraysum_align(incidence):
    if incidence == "P":
        return "P", 1
    return "SV", 3


def _water_level_spectral_div(num_fft, den_fft, water=0.01):
    scale = max(water * np.max(np.abs(den_fft)), 1e-30)
    den = den_fft.copy()
    den[np.abs(den) < scale] = scale + 0j
    return fftshift(np.real(ifft(num_fft / den)))


def spectral_rf_from_seis(radial, vertical, incidence, water=0.01):
    """Match PyRaysum spectral division (see pyraysum.prs.Result.calculate_rfs)."""
    ft_r, ft_z = fft(radial), fft(vertical)
    if incidence == "P":
        return _water_level_spectral_div(ft_r, ft_z, water)
    return _water_level_spectral_div(-ft_z, ft_r, water)


def compute_pyraysum_rf_spectral(spec, incidence):
    """Native PyRaysum receiver function (built-in spectral division)."""
    model, _, _ = build_pyraysum_model(spec)
    p_deg = P_DEG if incidence == "P" else S_DEG
    slow = rayp_deg_to_si(p_deg) * 1000.0
    wvtype, align = _pyraysum_align(incidence)
    rc = prs.Control(wvtype=wvtype, npts=NT, dt=DT, align=align, rot=1, mults=MULTS)
    result = prs.run(model, prs.Geometry(0.0, slow), rc, rf=True)
    rf_r = result.rfs[0][0]
    return rf_r.stats.taxis, rf_r.data.copy()


def compute_telewavesim_rf_spectral(spec, incidence):
    """Telewavesim seismograms + PyRaysum-style spectral division."""
    model_tws = build_tws_model(spec)
    p_deg = P_DEG if incidence == "P" else S_DEG
    slow = rayp_deg_to_si(p_deg) * 1000.0
    wvtype = "P" if incidence == "P" else "SV"
    st = tws_utils.run_plane(model_tws, slow, NT, DT, wvtype=wvtype)
    rf = spectral_rf_from_seis(-st[0].data, st[2].data, incidence)
    t_rf = (np.arange(NT) - NT // 2) * DT
    return t_rf, rf


def compute_pyraysum_rf(spec, incidence):
    """PyRaysum seismograms + seispy iterative RF (FK-compatible)."""
    model, rho, thick_m = build_pyraysum_model(spec)
    p_deg = P_DEG if incidence == "P" else S_DEG
    slow = rayp_deg_to_si(p_deg) * 1000.0
    wvtype = "P" if incidence == "P" else "SV"
    rc = prs.Control(wvtype=wvtype, npts=NT, dt=DT, align=0, rot=1, mults=MULTS)
    result = prs.run(model, prs.Geometry(0.0, slow), rc, rf=False)
    seis, _ = result[0]
    radial, vertical = _rtz_from_pyraysum_seis(seis)
    tshift = TSHIFT_P if incidence == "P" else TSHIFT_S
    decon_kwargs = dict(
        tshift=tshift, f0=F0_RF, itmax=ITMAX, minderr=MINDERR,
    )
    if incidence == "P":
        rfs, t_rf, rms_arr = compute_prf(
            radial[None, :], vertical[None, :], DT, **decon_kwargs,
        )
    else:
        rfs, t_rf, rms_arr = compute_srf(
            radial[None, :], vertical[None, :], DT, **decon_kwargs,
        )
    meta = {
        "package": "pyraysum",
        "incidence": incidence,
        "slow_km_s": slow,
        "p_deg": p_deg,
        "decon": "seispy_iter",
        "f0": F0_RF,
        "tshift": tshift,
        "align": 0,
        "mults": MULTS,
    }
    return t_rf, rfs[0], float(rms_arr[0]), meta, rho, thick_m


def compute_telewavesim_rf(spec, incidence):
    """Telewavesim seismograms + seispy iterative RF (FK-compatible)."""
    model_tws = build_tws_model(spec)
    p_deg = P_DEG if incidence == "P" else S_DEG
    slow = rayp_deg_to_si(p_deg) * 1000.0
    wvtype = "P" if incidence == "P" else "SV"
    st = tws_utils.run_plane(model_tws, slow, NT, DT, wvtype=wvtype)
    radial = -st[0].data.astype(np.float64)
    vertical = st[2].data.astype(np.float64)
    tshift = TSHIFT_P if incidence == "P" else TSHIFT_S
    decon_kwargs = dict(
        tshift=tshift, f0=F0_RF, itmax=ITMAX, minderr=MINDERR,
    )
    if incidence == "P":
        rfs, t_rf, rms_arr = compute_prf(
            radial[None, :], vertical[None, :], DT, **decon_kwargs,
        )
    else:
        rfs, t_rf, rms_arr = compute_srf(
            radial[None, :], vertical[None, :], DT, **decon_kwargs,
        )
    return t_rf, rfs[0], float(rms_arr[0])


def waveform_cc(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-30:
        return 0.0
    return float(np.max(np.correlate(a, b, mode="full")) / denom)


def cc_in_window(t_a, rf_a, t_b, rf_b, t_min, t_max):
    ma = (t_a >= t_min) & (t_a <= t_max)
    mb = (t_b >= t_min) & (t_b <= t_max)
    n = min(ma.sum(), mb.sum())
    if n < 10:
        return float("nan")
    return waveform_cc(rf_a[ma][:n], rf_b[mb][:n])


def plot_rf_compare(
    title,
    incidence,
    t_prs_sp,
    rf_prs_sp,
    t_tws_sp,
    rf_tws_sp,
    cc_sp,
    t_prs_it,
    rf_prs_it,
    t_tws_it,
    rf_tws_it,
    cc_it,
):
    """Two-panel figure: spectral (original) vs seispy iter (FK-compatible)."""
    tshift = TSHIFT_P if incidence == "P" else TSHIFT_S
    t_max = T_MAX_P if incidence == "P" else T_MAX_S
    color_prs, color_tws = "#0072B2", "#CC79A7"

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    ax = axes[0]
    ax.plot(t_prs_sp, rf_prs_sp, color=color_prs, lw=2.8, label="PyRaysum (spectral)")
    ax.plot(t_tws_sp, rf_tws_sp, color=color_tws, lw=1.8, label="Telewavesim (spectral)")
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlim(-tshift, t_max)
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{title} — {incidence}-wave RF: spectral division (original)")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.02, f"CC (PyRaysum vs TWS): {cc_sp:.4f}",
        transform=ax.transAxes, fontsize=10, va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax = axes[1]
    ax.plot(t_prs_it, rf_prs_it, color=color_prs, lw=2.8, label=f"PyRaysum (seispy f0={F0_RF})")
    ax.plot(t_tws_it, rf_tws_it, color=color_tws, lw=1.8, label=f"Telewavesim (seispy f0={F0_RF})")
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlim(-tshift, t_max)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{title} — {incidence}-wave RF: seispy iter (FK-compatible)")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.02, f"CC (PyRaysum vs TWS): {cc_it:.4f}",
        transform=ax.transAxes, fontsize=10, va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    return fig


def main():
    from seispy.decon import deconit  # noqa: F401

    print(f"pyraysum {prs.__version__}")
    print(f"PyRaysum mults={MULTS} (first-order free-surface multiples)")
    print(f"FK RF: seispy iter f0={F0_RF}; also saving spectral-division RFs for reference")
    combo = {}

    for mid, spec in MODELS.items():
        print("=" * 60, spec["title"])
        for incidence in ("P", "S"):
            tshift = TSHIFT_P if incidence == "P" else TSHIFT_S
            t_max = T_MAX_P if incidence == "P" else T_MAX_S
            t_min = -1.0

            t_prs_sp, rf_prs_sp = compute_pyraysum_rf_spectral(spec, incidence)
            t_tws_sp, rf_tws_sp = compute_telewavesim_rf_spectral(spec, incidence)
            cc_sp = cc_in_window(t_prs_sp, rf_prs_sp, t_tws_sp, rf_tws_sp, t_min, t_max)

            t_prs_it, rf_prs_it, rms, meta, rho, thick_m = compute_pyraysum_rf(spec, incidence)
            t_tws_it, rf_tws_it, rms_tws = compute_telewavesim_rf(spec, incidence)
            cc_it = cc_in_window(t_prs_it, rf_prs_it, t_tws_it, rf_tws_it, t_min, t_max)

            out = OUT_DIR / f"{mid}_{incidence}.npz"
            np.savez(
                out,
                model_id=mid,
                incidence=incidence,
                title=spec["title"],
                t_rf=t_prs_it,
                rf_radial=rf_prs_it,
                t_rf_tws=t_tws_it,
                rf_radial_tws=rf_tws_it,
                t_rf_spectral=t_prs_sp,
                rf_radial_spectral=rf_prs_sp,
                t_rf_spectral_tws=t_tws_sp,
                rf_radial_spectral_tws=rf_tws_sp,
                vp_km_s=np.array(spec["vp"]),
                vs_km_s=np.array(spec["vs"]),
                z_top_km=np.array(spec["z_top"]),
                thick_m=np.array(thick_m),
                rho=np.array(rho),
                p_deg=P_DEG if incidence == "P" else S_DEG,
                dt=DT,
                npts=NT,
                f0=F0_RF,
                tshift=tshift,
                decon_method="iter",
                rms=rms,
                rms_tws=rms_tws,
                cc_spectral=cc_sp,
                cc_seispy=cc_it,
                mults=MULTS,
                meta=json.dumps(meta),
            )
            combo[f"t_{mid}_{incidence}"] = t_prs_it
            combo[f"rf_{mid}_{incidence}"] = rf_prs_it

            print(
                f"  {incidence}-RF  spectral CC={cc_sp:.4f}  "
                f"seispy CC={cc_it:.4f}  RMS={rms:.4f}  saved {out.name}"
            )

            fig = plot_rf_compare(
                spec["title"], incidence,
                t_prs_sp, rf_prs_sp, t_tws_sp, rf_tws_sp, cc_sp,
                t_prs_it, rf_prs_it, t_tws_it, rf_tws_it, cc_it,
            )
            fig.savefig(FIG_DIR / f"{mid}_{incidence}_rf_compare.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(t_prs_it, rf_prs_it, color="#0072B2", lw=2.8, label="PyRaysum (seispy)")
            ax.plot(t_tws_it, rf_tws_it, color="#CC79A7", lw=1.8, label="Telewavesim (seispy)")
            ax.axvline(0, color="gray", ls="--", lw=0.8)
            ax.set_xlim(-tshift, t_max)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"{spec['title']} — {incidence}-wave RF (seispy f0={F0_RF})")
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            ax.text(0.02, 0.02, f"CC: {cc_it:.4f}", transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"{mid}_{incidence}_rf_seispy.png", dpi=150)
            plt.close(fig)

    np.savez_compressed(OUT_DIR / "pyraysum_rfs_all.npz", **combo)
    print(f"\nSaved combined archive: {OUT_DIR / 'pyraysum_rfs_all.npz'}")
    print(f"Figures: {FIG_DIR}  (*_rf_compare.png = spectral + seispy panels)")


if __name__ == "__main__":
    main()
