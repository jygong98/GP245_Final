"""Readers for the original FDFK2D Fortran input/output formats.

- ASCII parameter decks: ``FD_model.dat``, ``Source.dat``, ``Receiver.dat``,
  ``FK_model_{left,right}.dat``.
- Seismic Unix (SU) binary traces: 240-byte header + ``ns`` float32 samples,
  little-endian, with ``ns`` at int16 index 57 and sample interval
  (microseconds) at int16 index 58 (matching Submain.f90).

Performance (see ``notebooks/0_io_test.ipynb``): **NPY** and **vectorized SU**
(``read_su`` / ``write_su``) are the fastest read/write paths; HDF5 (LZF) is
the slowest among formats tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import backazimuth
from .model import FKLayerModel
from .source import RickerSource


# --------------------------------------------------------------------------- #
# Token parsing helpers
# --------------------------------------------------------------------------- #
def fortran_float(token: str) -> float:
    """Parse a Fortran float token, handling ``d``/``D`` exponents (e.g. ``35.d3``)."""
    return float(token.strip().replace("d", "e").replace("D", "e"))


def fortran_bool(token: str) -> bool:
    """Parse a Fortran logical token (``.true.`` / ``.false.``)."""
    t = token.strip().lower()
    if t.startswith(".t") or t == "t" or t == "true":
        return True
    if t.startswith(".f") or t == "f" or t == "false":
        return False
    raise ValueError(f"Cannot parse Fortran logical: {token!r}")


def _data_lines(path: Path) -> list[str]:
    """Return non-empty, stripped lines of a text file."""
    text = Path(path).read_text()
    return [ln.strip() for ln in text.splitlines() if ln.strip() != ""]


# --------------------------------------------------------------------------- #
# Parameter decks
# --------------------------------------------------------------------------- #
@dataclass
class FDModelParams:
    """Parsed ``FD_model.dat`` (geometry/time grid + geographic reference).

    ``norder`` is the **half-order** of the spatial FD operator (number of
    grid points on each side of the stencil centre).  ``FD_model.dat``
    stores the full order (``fd_order``); :func:`read_fd_model` converts
    it via ``norder = fd_order // 2``, matching Fortran's
    ``norder = int(norder/2)`` in ``Input.f90``.
    """

    norder: int
    x0: float
    xn: float
    z0: float
    zn: float
    dx: float
    dz: float
    dt: float
    tmax: float
    tstep: float
    npml: int
    is_pml_top: bool
    lat_org: float
    lon_org: float
    az_org: float
    outsnap: bool
    nstep: int

    @property
    def profile_length(self) -> float:
        return self.xn - self.x0

    @property
    def x_center(self) -> float:
        return 0.5 * (self.x0 + self.xn)

    @property
    def nt(self) -> int:
        """Number of time samples (inclusive of t=0), matching ns in the SU files."""
        return int(round(self.tmax / self.dt)) + 1


@dataclass
class SourceParams:
    """Parsed ``Source.dat``.

    ``evla`` and ``evlo`` are the event latitude and longitude (degrees).
    They are not used in the FK propagator itself; :func:`load_example` passes
    them to :func:`geometry.backazimuth` together with ``FD_model.dat`` reference
    ``(lat_org, lon_org)`` to obtain the back-azimuth relative to the profile
    (``phi = baz - az_org``). That angle sets the along-profile horizontal
    slowness ``p_along = -p * cos(phi)`` in :func:`engine.compute_fk_seismograms`.
    If the profile is aligned with the wave (or you call the engine with an
    explicit ``phi``), these fields have no effect on the result.
    """

    f0: float
    strength: float
    is_p: bool
    rayp_deg: float
    evla: float
    evlo: float

    @property
    def incidence(self) -> str:
        return "P" if self.is_p else "S"


def read_fd_model(path) -> FDModelParams:
    """Parse an ``FD_model.dat`` file.

    The deck alternates a comment line and a value line; comment lines are
    those that do not start with a digit, sign, dot or Fortran logical.
    """
    lines = _data_lines(path)
    values = [ln for ln in lines if ln[0] in "+-." or ln[0].isdigit()
              or ln.lower().startswith(".t") or ln.lower().startswith(".f")]
    # Expected value lines, in order.
    fd_order = int(float(values[0].split()[0]))   # full FD order from file
    norder = fd_order // 2                        # half-order for stencil loops
    g = values[1].split()
    x0, xn, z0, zn, dx, dz, dt = (fortran_float(t) for t in g[:7])
    t_line = values[2].split()
    tmax = fortran_float(t_line[0])
    tstep = fortran_float(t_line[1])
    npml = int(float(t_line[2]))
    # Fortran: if (2*norder > (npml+1)) norder = int((npml+1)/2)
    if 2 * norder > npml + 1:
        norder = (npml + 1) // 2
    is_pml_top = fortran_bool(t_line[3])
    geo = values[3].split()
    lat_org, lon_org, az_org = (fortran_float(t) for t in geo[:3])
    snap = values[4].split()
    outsnap = fortran_bool(snap[0])
    nstep = int(float(snap[1]))
    return FDModelParams(
        norder=norder, x0=x0, xn=xn, z0=z0, zn=zn, dx=dx, dz=dz, dt=dt,
        tmax=tmax, tstep=tstep, npml=npml, is_pml_top=is_pml_top,
        lat_org=lat_org, lon_org=lon_org, az_org=az_org,
        outsnap=outsnap, nstep=nstep,
    )


def read_source(path) -> SourceParams:
    """Parse a ``Source.dat`` file."""
    lines = _data_lines(path)
    values = [ln for ln in lines if ln[0] in "+-." or ln[0].isdigit()
              or ln.lower().startswith(".t") or ln.lower().startswith(".f")]
    a = values[0].split()
    f0 = fortran_float(a[0])
    strength = fortran_float(a[1])
    is_p = fortran_bool(a[2])
    b = values[1].split()
    rayp_deg = fortran_float(b[0])
    evla = fortran_float(b[1])
    evlo = fortran_float(b[2])
    return SourceParams(
        f0=f0, strength=strength, is_p=is_p,
        rayp_deg=rayp_deg, evla=evla, evlo=evlo,
    )


def read_receivers(path):
    """Parse a ``Receiver.dat`` file. Returns ``(rx, rz)`` arrays (m)."""
    lines = _data_lines(path)
    nrcv = int(float(lines[1].split()[0]))
    rx, rz = [], []
    for ln in lines[2:]:
        parts = ln.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            z = float(parts[1])
        except ValueError:
            continue  # skip the "rx(m) rz(m)" header line
        rx.append(x)
        rz.append(z)
        if len(rx) == nrcv:
            break
    return np.asarray(rx), np.asarray(rz)


def read_fk_model(path, rho=None) -> FKLayerModel:
    """Parse an ``FK_model_*.dat`` file into an :class:`FKLayerModel`.

    ``nlayer`` counts all rows including the half-space (the last row). Only
    the first ``nlayer`` data rows are used (extra rows are ignored, matching
    the Fortran reader). Density defaults to :func:`density_from_vp_fdfk`.
    """
    lines = _data_lines(path)
    nlayer = int(float(lines[1].split()[0]))
    rows = []
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            vp = fortran_float(parts[0])
            vs = fortran_float(parts[1])
            zt = fortran_float(parts[2])
        except ValueError:
            continue  # header line "Vp Vs z+"
        rows.append((vp, vs, zt))
        if len(rows) == nlayer:
            break
    if len(rows) < nlayer:
        raise ValueError(f"{path}: expected {nlayer} layer rows, got {len(rows)}.")
    vp = np.array([r[0] for r in rows])
    vs = np.array([r[1] for r in rows])
    zt = np.array([r[2] for r in rows])
    return FKLayerModel.from_layers(vp, vs, zt, rho=rho)


# --------------------------------------------------------------------------- #
# SU binary I/O
# --------------------------------------------------------------------------- #
_SU_HDR_BYTES = 240
_SU_NS_IDX = 57   # int16 index of number-of-samples
_SU_DT_IDX = 58   # int16 index of sample interval (microseconds)


def _read_su_loop(path):
    """Original loop-based SU reader (kept for benchmarking)."""
    raw = Path(path).read_bytes()
    hdr0 = np.frombuffer(raw[:_SU_HDR_BYTES], dtype="<i2")
    ns = int(hdr0[_SU_NS_IDX])
    dt_us = int(hdr0[_SU_DT_IDX])
    if ns <= 0:
        raise ValueError(f"{path}: invalid ns={ns} in SU header.")
    trace_bytes = _SU_HDR_BYTES + ns * 4
    if len(raw) % trace_bytes != 0:
        raise ValueError(f"{path}: file size not a multiple of trace size.")
    ntr = len(raw) // trace_bytes
    data = np.empty((ntr, ns), dtype=np.float32)
    for i in range(ntr):
        off = i * trace_bytes + _SU_HDR_BYTES
        data[i] = np.frombuffer(raw[off:off + ns * 4], dtype="<f4")
    return data, dt_us * 1e-6


def read_su(path):
    """Read an SU file (vectorized). Returns ``(data, dt_sec)`` with shape ``(ntr, ns)``."""
    raw = np.fromfile(Path(path), dtype=np.uint8)
    if raw.size < _SU_HDR_BYTES:
        raise ValueError(f"{path}: file too small for an SU header.")
    hdr0 = raw[:_SU_HDR_BYTES].view("<i2")
    ns = int(hdr0[_SU_NS_IDX])
    dt_us = int(hdr0[_SU_DT_IDX])
    if ns <= 0:
        raise ValueError(f"{path}: invalid ns={ns} in SU header.")
    trace_bytes = _SU_HDR_BYTES + ns * 4
    if raw.size % trace_bytes != 0:
        raise ValueError(f"{path}: file size not a multiple of trace size.")
    ntr = raw.size // trace_bytes
    traces = raw[: ntr * trace_bytes].reshape(ntr, trace_bytes)
    data = traces[:, _SU_HDR_BYTES:].copy().view("<f4").reshape(ntr, ns)
    return data, dt_us * 1e-6


def _write_su_loop(path, data, dt_sec):
    """Original loop-based SU writer (kept for benchmarking)."""
    data = np.atleast_2d(np.asarray(data, dtype=np.float32))
    ntr, ns = data.shape
    dt_us = int(round(dt_sec * 1e6))
    out = bytearray()
    for i in range(ntr):
        hdr = np.zeros(120, dtype="<i2")
        hdr[_SU_NS_IDX] = ns
        hdr[_SU_DT_IDX] = dt_us
        out += hdr.tobytes()
        out += np.ascontiguousarray(data[i], dtype="<f4").tobytes()
    Path(path).write_bytes(bytes(out))


def write_su(path, data, dt_sec):
    """Write traces to an SU file (vectorized).

    Parameters
    ----------
    path : path-like
    data : array_like, shape (ntr, ns)
    dt_sec : float
        Sample interval in seconds (stored as microseconds in the header).
    """
    data = np.atleast_2d(np.asarray(data, dtype=np.float32))
    ntr, ns = data.shape
    dt_us = int(round(dt_sec * 1e6))
    trace_bytes = _SU_HDR_BYTES + ns * 4
    buf = np.zeros((ntr, trace_bytes), dtype=np.uint8)
    hdr_template = np.zeros(_SU_HDR_BYTES, dtype=np.uint8)
    hdr_view = hdr_template.view("<i2")
    hdr_view[_SU_NS_IDX] = ns
    hdr_view[_SU_DT_IDX] = dt_us
    buf[:, :_SU_HDR_BYTES] = hdr_template
    buf[:, _SU_HDR_BYTES:] = np.ascontiguousarray(data).view(np.uint8).reshape(
        ntr, ns * 4
    )
    buf.tofile(str(Path(path)))


# --------------------------------------------------------------------------- #
# NumPy .npz I/O
# --------------------------------------------------------------------------- #
def write_npz(path, data, dt_sec, **extra):
    """Write traces to a compressed ``.npz`` file.

    Stores ``data`` (float32) and ``dt`` as arrays inside the archive.
    Additional keyword arguments are saved as extra arrays.
    """
    data = np.atleast_2d(np.asarray(data, dtype=np.float32))
    np.savez(str(Path(path)), data=data, dt=np.float64(dt_sec), **extra)


def read_npz(path):
    """Read a ``.npz`` trace file. Returns ``(data, dt_sec)``."""
    with np.load(str(Path(path))) as npz:
        data = npz["data"]
        dt_sec = float(npz["dt"])
    return data, dt_sec


def write_npy(path, data, dt_sec):
    """Write traces as a raw ``.npy`` file (fastest possible write).

    Metadata (dt) is stored in a companion ``<stem>_dt.npy`` file.
    """
    path = Path(path)
    data = np.atleast_2d(np.asarray(data, dtype=np.float32))
    np.save(str(path), data)
    np.save(str(path.with_name(path.stem + "_dt.npy")), np.float64(dt_sec))


def read_npy(path, mmap_mode=None):
    """Read a ``.npy`` trace file. Returns ``(data, dt_sec)``.

    Parameters
    ----------
    mmap_mode : str or None
        Pass ``'r'`` for memory-mapped read (zero-copy, near-instant).
    """
    path = Path(path)
    data = np.load(str(path), mmap_mode=mmap_mode)
    dt_path = path.with_name(path.stem + "_dt.npy")
    dt_sec = float(np.load(str(dt_path)))
    return data, dt_sec


# --------------------------------------------------------------------------- #
# HDF5 I/O (optional dependency)
# --------------------------------------------------------------------------- #
def _import_h5py():
    try:
        import h5py
        return h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 I/O. Install with: pip install h5py"
        )


def write_hdf5(path, data, dt_sec, compression="lzf", dataset="data", **attrs):
    """Write traces to an HDF5 file with optional compression.

    Parameters
    ----------
    path : path-like
    data : array_like, shape (ntr, ns)
    dt_sec : float
    compression : str or None
        Compression filter (``'lzf'``, ``'gzip'``, or ``None``).
    dataset : str
        Name of the dataset inside the HDF5 file.
    **attrs : dict
        Extra attributes stored on the dataset.
    """
    h5py = _import_h5py()
    data = np.atleast_2d(np.asarray(data, dtype=np.float32))
    with h5py.File(str(Path(path)), "w") as f:
        ds = f.create_dataset(dataset, data=data, compression=compression)
        ds.attrs["dt"] = dt_sec
        for k, v in attrs.items():
            ds.attrs[k] = v


def read_hdf5(path, dataset="data"):
    """Read traces from an HDF5 file. Returns ``(data, dt_sec)``."""
    h5py = _import_h5py()
    with h5py.File(str(Path(path)), "r") as f:
        ds = f[dataset]
        data = ds[:]
        dt_sec = float(ds.attrs["dt"])
    return data, dt_sec


# --------------------------------------------------------------------------- #
# Unified trace reader (auto-detect format by extension)
# --------------------------------------------------------------------------- #
def read_traces(path):
    """Auto-detect file format and read traces. Returns ``(data, dt_sec)``.

    Supported extensions: ``.su``, ``.npz``, ``.npy``, ``.h5``, ``.hdf5``.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".su":
        return read_su(path)
    elif ext == ".npz":
        return read_npz(path)
    elif ext == ".npy":
        return read_npy(path)
    elif ext in (".h5", ".hdf5"):
        return read_hdf5(path)
    else:
        raise ValueError(
            f"Unsupported file extension {ext!r}. "
            f"Expected one of: .su, .npz, .npy, .h5, .hdf5"
        )


# --------------------------------------------------------------------------- #
# Convenience bundle for running an example input directory
# --------------------------------------------------------------------------- #
@dataclass
class FKExample:
    """Everything needed to run the FK for one example input directory."""

    model: FKLayerModel
    source: RickerSource
    fd: FDModelParams
    src: SourceParams
    rx: np.ndarray
    rz: np.ndarray
    phi: float          # baz - az_org: angle between back-azimuth and profile (rad)
    x0_plane: float     # plane-wave reference x (m)

    @property
    def dt(self) -> float:
        return self.fd.dt

    @property
    def nt(self) -> int:
        return self.fd.nt


def load_example(input_dir, side: str = "left") -> FKExample:
    """Load an FDFK2D example input directory into an :class:`FKExample`.

    ``side`` selects ``FK_model_left.dat`` or ``FK_model_right.dat``.
    The back-azimuth (relative to the profile) is computed from the source
    location and the profile reference point ``(lat_org, lon_org)``, which is an
    excellent approximation for a limited-aperture array at teleseismic range.
    """
    input_dir = Path(input_dir)
    fd = read_fd_model(input_dir / "FD_model.dat")
    src = read_source(input_dir / "Source.dat")
    rx, rz = read_receivers(input_dir / "Receiver.dat")
    fk_file = f"FK_model_{side}.dat"
    model = read_fk_model(input_dir / fk_file)

    source = RickerSource(
        f0=src.f0, strength=src.strength, incidence=src.incidence,
        t0=1.2 / src.f0,
    )

    baz = backazimuth(src.evla, src.evlo, fd.lat_org, fd.lon_org)
    phi = np.deg2rad(baz - fd.az_org)

    return FKExample(
        model=model, source=source, fd=fd, src=src,
        rx=rx, rz=rz, phi=phi, x0_plane=fd.x0,
    )
