"""Plotting utilities for receiver functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rf_power_law(trace: np.ndarray, power: float) -> np.ndarray:
    """Sign-preserving power-law transform on raw RF amplitudes."""
    if power <= 0:
        raise ValueError(f"power must be positive, got {power}")
    return np.sign(trace) * (np.abs(trace) ** power)


def rf_amp_percentile(
    *rfs_pow: np.ndarray,
    mask: np.ndarray,
    percentile: float = 80.0,
) -> float:
    """Pooled amplitude reference from post-power |RF| samples.

    Returns the given percentile of absolute amplitudes across all supplied
    RF arrays within ``mask`` (typically the post-arrival window).
    """
    if not 0 < percentile <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    if not rfs_pow:
        raise ValueError("at least one RF array is required")
    vals = np.concatenate([np.abs(rf[:, mask]).ravel() for rf in rfs_pow])
    ref = float(np.percentile(vals, percentile))
    return ref if ref > 1e-30 else 1e-30


def sac_power_wiggle(
    trace: np.ndarray,
    amp_ref: float,
    bin_size_km: float,
    *,
    power: float = 2.0,
) -> np.ndarray:
    """Power-law wiggle then shared normalization (GMT/SAC style).

    Operations are applied in this order:

        1. powered = sign(x) * |x|^power
        2. wiggle  = powered / amp_ref * bin_size_km

    ``amp_ref`` is typically a high percentile (e.g. 80%) of ``|powered|``
    in the post-arrival reference window, computed after the power transform.

    With ``power > 1``, strong phases are emphasized before normalization;
    ``power = 1`` recovers linear scaling.
    """
    if amp_ref <= 0:
        raise ValueError(f"amp_ref must be positive, got {amp_ref}")
    powered = rf_power_law(trace, power)
    return powered / amp_ref * bin_size_km


def plot_single_rf(
    t_rf: np.ndarray,
    rf: np.ndarray,
    out_png: Path,
    *,
    title: str = "Receiver Function",
) -> None:
    """Plot a single receiver function trace."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_rf, rf, "k", lw=1.0)
    ax.fill_between(t_rf, rf, where=(rf > 0), color="red", alpha=0.4)
    ax.fill_between(t_rf, rf, where=(rf < 0), color="blue", alpha=0.4)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(t_rf[0], t_rf[-1])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_rf_section(
    t_rf: np.ndarray,
    rfs: np.ndarray,
    out_png: Path,
    *,
    title: str = "RF Section",
    decimate: int = 1,
    t_max: float | None = None,
) -> None:
    """Plot receiver function section (wiggle traces stacked vertically)."""
    nrcv = rfs.shape[0]
    indices = np.arange(0, nrcv, decimate)

    fig, ax = plt.subplots(figsize=(8, max(6, len(indices) * 0.08)))
    norm = np.max(np.abs(rfs)) + 1e-30
    scale = decimate * 0.8

    for j, idx in enumerate(indices):
        tr = rfs[idx] / norm * scale
        ax.plot(t_rf, j + tr, "k", lw=0.5)
        ax.fill_between(t_rf, j + tr, j, where=(tr > 0), color="red", alpha=0.3)
        ax.fill_between(t_rf, j + tr, j, where=(tr < 0), color="blue", alpha=0.3)

    ax.axvline(0, color="gray", ls="--", lw=0.8, label="P arrival")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trace index")
    ax.set_title(title)
    ax.set_yticks(np.arange(0, len(indices), max(1, len(indices) // 10)))
    ax.set_yticklabels(indices[::max(1, len(indices) // 10)])
    if t_max is not None:
        ax.set_xlim(t_rf[0], t_max)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_rf_stack(
    t_rf: np.ndarray,
    rfs: np.ndarray,
    out_png: Path,
    *,
    title: str = "Stacked RF",
) -> None:
    """Plot the mean (stacked) receiver function."""
    stack = np.mean(rfs, axis=0)
    plot_single_rf(t_rf, stack, out_png, title=title)
