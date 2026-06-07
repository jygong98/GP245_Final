# Hybrid FD-FK Module — API Reference

> Auto-generated from source code. Main entry point: `hybrid.engine.compute_hybrid_seismograms`.

---

## Quick Start

```python
from fk.model import FKLayerModel
from fk.source import RickerSource
from fk.geometry import rayp_deg_to_si
from hybrid import HybridConfig, compute_hybrid_seismograms

config = HybridConfig(nx=101, nz=61, dx=500.0, dz=500.0, dt=0.02, nt=1001)
model = FKLayerModel.from_layers(vp=[6000.0], vs=[3464.0], z_top=[0.0])
source = RickerSource(f0=0.5, incidence="P")

result = compute_hybrid_seismograms(
    config, model, model, source,
    p_si=rayp_deg_to_si(4.0),
    receivers_x=[25000.0],
)
# result.ux  -> shape (1, 1001)  horizontal displacement
# result.uz  -> shape (1, 1001)  vertical displacement
```

---

## 1. `compute_hybrid_seismograms` — Single-Run API

**Location:** `hybrid/engine.py`

```python
def compute_hybrid_seismograms(
    config, model_fk_left, model_fk_right, source, p_si, receivers_x,
    receivers_z=None, *, vp_2d=None, vs_2d=None, rho_2d=None,
    phi=0.0, z_init=None, freq_damping=0.0, fortran_seisx_convention=True,
    verbose=True, snapshot_interval=0,
) -> HybridResult
```

### 1.1 Required Parameters

| Parameter | Type | Description |
|---|---|---|
| `config` | [`HybridConfig`](#3-hybridconfig) | Grid, time-stepping, FD stencil, and PML settings. |
| `model_fk_left` | [`FKLayerModel`](#4-fklayermodel) | 1D layered velocity model for the **left** FK boundary. |
| `model_fk_right` | [`FKLayerModel`](#4-fklayermodel) | 1D model for the **right** FK boundary. For laterally homogeneous models, pass the same object for both. |
| `source` | [`RickerSource`](#5-rickersource) | Ricker wavelet and P/S incidence type. |
| `p_si` | `float` | Horizontal slowness in **s/m**. Use `fk.geometry.rayp_deg_to_si(p_deg)` to convert from s/deg. |
| `receivers_x` | `array_like` | Receiver x-coordinates (m) within the FD domain. |

### 1.2 Optional Parameters (keyword-only)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `receivers_z` | `array_like` | `None` → surface (z=0) | Receiver z-coordinates (m). |
| `vp_2d` | `ndarray (nz, nx)` | `None` | 2D P-velocity for the FD domain. If `None`, built from `model_fk_left` (laterally homogeneous). |
| `vs_2d` | `ndarray (nz, nx)` | `None` | 2D S-velocity for the FD domain. |
| `rho_2d` | `ndarray (nz, nx)` | `None` | 2D density for the FD domain. |
| `phi` | `float` | `0.0` | Back-azimuth angle (rad) for plane-wave horizontal moveout. |
| `z_init` | `float` | auto | Plane-wave initialization depth (m) in the half-space. Default: `zn + 0.5 * profile_length * tan(incidence_angle)`. |
| `freq_damping` | `float` | `0.0` | FK imaginary-frequency stabilizer (rad/s). `0` disables. |
| `fortran_seisx_convention` | `bool` | `True` | Apply an extra `cos(phi)` to recorded `ux` so output matches Fortran `seisx.su` (double-cos convention). Set `False` for the single-cos profile-projected ux. `uz` is unaffected. No-op when `phi=0`. |
| `verbose` | `bool` | `True` | Print progress messages. |
| `snapshot_interval` | `int` | `0` | If > 0, store full wavefield snapshots every N steps (debug mode, slower). |

### 1.3 Returns → [`HybridResult`](#6-hybridresult)

---

## 2. `HybridEngine` — Batch API (Caches FK Between Runs)

**Location:** `hybrid/engine.py`

Pre-computes FK Green's functions once; reuses them across multiple forward runs with varying 2D models or source wavelets.

### 2.1 Constructor: `HybridEngine.__init__`

```python
HybridEngine(
    config, model_fk_left, model_fk_right, source, p_si,
    *, phi=0.0, z_init=None, freq_damping=0.0, fortran_seisx_convention=True,
    verbose=True,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `HybridConfig` | — | Grid, time, PML settings. |
| `model_fk_left` | `FKLayerModel` | — | Left boundary 1D model. |
| `model_fk_right` | `FKLayerModel` | — | Right boundary 1D model. |
| `source` | `RickerSource` | — | Initial source wavelet. |
| `p_si` | `float` | — | Horizontal slowness (s/m). |
| `phi` | `float` | `0.0` | Back-azimuth (rad). |
| `z_init` | `float` | auto | Plane-wave init depth (m). |
| `freq_damping` | `float` | `0.0` | FK frequency damping (rad/s). |
| `fortran_seisx_convention` | `bool` | `True` | Extra `cos(phi)` on output `ux` for Fortran `seisx.su` alignment. |
| `verbose` | `bool` | `True` | Print setup messages. |

### 2.2 `HybridEngine.run`

```python
engine.run(receivers_x, receivers_z=None, *, vp_2d=None, vs_2d=None, rho_2d=None) -> HybridResult
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `receivers_x` | `array_like` | — | Receiver x-coordinates (m). |
| `receivers_z` | `array_like` | `None` → surface | Receiver z-coordinates (m). |
| `vp_2d` | `ndarray (nz, nx)` | `None` | Override 2D P-velocity for this run. |
| `vs_2d` | `ndarray (nz, nx)` | `None` | Override 2D S-velocity for this run. |
| `rho_2d` | `ndarray (nz, nx)` | `None` | Override 2D density for this run. |

**Returns:** [`HybridResult`](#6-hybridresult)

### 2.3 `HybridEngine.update_source`

```python
engine.update_source(source: RickerSource) -> None
```

Rebuilds boundary wavefield for a new source wavelet. Only performs cheap spectrum multiplication + IFFT; FK propagator matrices are reused.

---

## 3. `HybridConfig`

**Location:** `hybrid/config.py`

### 3.1 Fields (constructor parameters)

| Field | Type | Default | Description |
|---|---|---|---|
| `nx` | `int` | *(required)* | Number of interior grid points in x. |
| `nz` | `int` | *(required)* | Number of interior grid points in z. |
| `dx` | `float` | *(required)* | Grid spacing in x (m). |
| `dz` | `float` | *(required)* | Grid spacing in z (m). |
| `x0` | `float` | `0.0` | Domain origin x (m). |
| `z0` | `float` | `0.0` | Domain origin z (m). |
| `dt` | `float` | `0.01` | Time step (s). Must satisfy CFL stability. |
| `nt` | `int` | `4501` | Total number of time steps. |
| `norder` | `int` | `3` | Half-order of spatial FD stencil. Full order = `2 * norder` (default → 6th-order). |
| `npml` | `int` | `20` | PML absorbing-boundary grid points on left/right/bottom edges. |

### 3.2 Derived Properties (read-only)

| Property | Type | Description |
|---|---|---|
| `nbd` | `int` | Coupling-zone width = `2 * norder - 1` grid points. |
| `xn` | `float` | Right edge of domain (m) = `x0 + (nx-1) * dx`. |
| `zn` | `float` | Bottom edge of domain (m) = `z0 + (nz-1) * dz`. |
| `x_coords` | `ndarray (nx,)` | Interior x-coordinates (m). |
| `z_coords` | `ndarray (nz,)` | Interior z-coordinates (m). |
| `tmax` | `float` | Total simulation time (s) = `(nt-1) * dt`. |

---

## 4. `FKLayerModel`

**Location:** `fk/model.py`

1D layered isotropic elastic model. Last entry = half-space.

### 4.1 Fields

| Field | Type | Description |
|---|---|---|
| `vp` | `ndarray` | P-wave velocity (m/s), one value per layer + half-space. |
| `vs` | `ndarray` | S-wave velocity (m/s). |
| `rho` | `ndarray` | Density (kg/m³). |
| `z_top` | `ndarray` | Top depth of each layer (m), strictly increasing. |

### 4.2 Derived (auto-computed in `__post_init__`)

| Field | Type | Description |
|---|---|---|
| `mu` | `ndarray` | Shear modulus = `rho * vs²`. |
| `lam` | `ndarray` | Lamé's first parameter = `rho * (vp² - 2 vs²)`. |

### 4.3 Constructor Shortcut

```python
FKLayerModel.from_layers(vp, vs, z_top, rho=None)
```

If `rho` is `None`, uses FDFK2D linear relation via `density_from_vp_fdfk`: `rho = (vp + 980) / 2.760` (vp in m/s → rho in kg/m³).

### 4.4 Properties

| Property | Type | Description |
|---|---|---|
| `n_entries` | `int` | Total number of entries (layers + half-space). |
| `n_finite_layers` | `int` | Entries above the half-space. |
| `thickness` | `ndarray` | Thickness of each layer (m); half-space = `inf`. |
| `halfspace_index` | `int` | Index of the half-space entry (last). |

---

## 5. `RickerSource`

**Location:** `fk/source.py`

### 5.1 Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `f0` | `float` | *(required)* | Dominant frequency (Hz). Must be > 0. |
| `strength` | `float` | `1.0` | Amplitude scale (`ds` in original FDFK2D). |
| `incidence` | `str` | `"P"` | `"P"` or `"S"` incident plane wave. |
| `t0` | `float` | `1.0 / f0` | Ricker peak time (s). |

### 5.2 Methods

| Method | Signature | Returns |
|---|---|---|
| `wavelet` | `wavelet(t: ndarray) -> ndarray` | Ricker wavelet: `(1 - 2(πf₀(t-t₀))²) exp(-(πf₀(t-t₀))²)` scaled by `strength`. |
| `spectrum` | `spectrum(nt: int, dt: float) -> ndarray` | One-sided rfft spectrum, length `nt//2 + 1`. |

---

## 6. `HybridResult`

**Location:** `hybrid/engine.py`

### 6.1 Standard Fields (always present)

| Field | Type | Shape | Description |
|---|---|---|---|
| `t` | `ndarray` | `(nt,)` | Time axis (s). |
| `ux` | `ndarray` | `(nrcv, nt)` | Horizontal (radial) displacement seismograms. `float32`. |
| `uz` | `ndarray` | `(nrcv, nt)` | Vertical displacement seismograms. `float32`. |
| `receivers_x` | `ndarray` | `(nrcv,)` | Receiver x-coordinates (m). |
| `receivers_z` | `ndarray` | `(nrcv,)` | Receiver z-coordinates (m). |
| `config` | `HybridConfig` | — | Configuration used for this run. |

### 6.2 Debug Fields (only when `snapshot_interval > 0`)

| Field | Type | Description |
|---|---|---|
| `snapshots_ux` | `list[ndarray]` | Full-domain ux snapshots, each `(nz, nx)`. |
| `snapshots_uz` | `list[ndarray]` | Full-domain uz snapshots, each `(nz, nx)`. |

---

## 7. Lower-Level APIs

### 7.1 `compute_fk_greens` — Source-Independent FK Pre-Computation

**Location:** `hybrid/background.py`

```python
compute_fk_greens(
    config, model_left, model_right, p_si, incidence,
    *, phi=0.0, z_init=None,
) -> FKGreensCache
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `HybridConfig` | — | Grid and simulation parameters. |
| `model_left` | `FKLayerModel` | — | Left boundary 1D model. |
| `model_right` | `FKLayerModel` | — | Right boundary 1D model. |
| `p_si` | `float` | — | Horizontal slowness (s/m). |
| `incidence` | `str` | — | `"P"` or `"S"`. |
| `phi` | `float` | `0.0` | Back-azimuth (rad). |
| `z_init` | `float` | auto | Plane-wave init depth (m). |

**Returns:** `FKGreensCache` (opaque; pass to `boundary_wavefield_from_greens`).

### 7.2 `boundary_wavefield_from_greens` — Apply Source to Cached Greens

```python
boundary_wavefield_from_greens(
    config, greens, source, *, freq_damping=0.0,
) -> BoundaryWavefield
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `HybridConfig` | — | Grid and simulation parameters. |
| `greens` | `FKGreensCache` | — | Cached FK Green's functions. |
| `source` | `RickerSource` | — | Source wavelet. |
| `freq_damping` | `float` | `0.0` | Imaginary-frequency stabilizer (rad/s). |

**Returns:** `BoundaryWavefield` with fields:

| Field | Shape | Description |
|---|---|---|
| `ux_left`, `uz_left` | `(nz, 2*nbd, nt)` | Background displacement at left coupling zone. |
| `ux_right`, `uz_right` | `(nz, 2*nbd, nt)` | Right coupling zone. |
| `ux_bottom`, `uz_bottom` | `(nx, 2*nbd, nt)` | Bottom coupling zone. |

### 7.3 `compute_boundary_wavefield` — One-Shot (Source-Dependent)

```python
compute_boundary_wavefield(
    config, model_left, model_right, source, p_si,
    phi=0.0, z_init=None, freq_damping=0.0,
) -> BoundaryWavefield
```

Combines `compute_fk_greens` + `boundary_wavefield_from_greens` in a single call. Convenient for single-run usage but does not cache Green's functions.

---

## 8. Utility: Ray-Parameter Conversion

**Location:** `fk/geometry.py`

| Function | Signature | Description |
|---|---|---|
| `rayp_deg_to_si` | `(p_deg: float) -> float` | s/deg → s/m. Formula: `p_deg * 180 / (π × 6371000)`. |
| `rayp_si_to_deg` | `(p_si: float) -> float` | s/m → s/deg. |

---

## 9. Units Convention

All quantities use **SI units**:

| Quantity | Unit |
|---|---|
| Distance / coordinates | m |
| Velocity | m/s |
| Density | kg/m³ |
| Time / dt | s |
| Frequency | Hz |
| Slowness (`p_si`) | s/m |
| Angles (`phi`) | rad |
| Displacement output | m (relative to `strength`) |
