# FDFK_Python

**Python implementation of FDFK2D** — teleseismic plane-wave modeling and receiver-function (RF) analysis.

This project reproduces the **FDFK2D hybrid method** (finite-difference FD + wavenumber FK) from Liu et al. (2025, *SRL*):

| Module | Description |
|--------|-------------|
| `fk/` | FK / propagator-matrix method in 1D layered media; computes surface `(u_x, u_z)` seismograms |
| `hybrid/` | FD–FK coupling (TF/SF wavefield injection), 2D hybrid forward modeling |
| `rf/` | P/S receiver functions from FK or hybrid seismograms |
| `plots/` | Wavefield snapshots, animations, and other visualization tools |

The original FDFK2D FK core is a closed-source library; this repository reimplements it from scratch using paper Appendix A and validates against Fortran reference output and `benchmark/` data.

---

## Installation

**Recommended (course environment):**

```bash
conda activate GP245_Final
pip install -r requirements.txt   # if dependencies are not yet installed
```

**One-command standalone environment:**

```bash
bash scripts/setup_env.sh         # create/update conda env fdfk and run smoke tests
# or
conda env create -f environment-minimal.yml && conda activate fdfk
```

See `requirements.txt` / `environment-minimal.yml` for dependencies (NumPy, SciPy, Numba, Matplotlib, PyYAML, python-seispy, etc.).

---

## `examples/` directory

Top-level example subdirectories (each with `run_workflow.py`, a Jupyter workflow, and README):

| Directory | Contents |
|-----------|----------|
| `0_config_examples/` | YAML configuration templates (FK / hybrid / Antarctic ice cover, etc.) |
| `_common/` | Shared example workflow script `rf_example_workflow.py` |
| `1_Crust_Mantle_P/` | Single-layer crust over mantle, P-wave incidence (`benchmark/test1_P_out`) |
| `1_Crust_Mantle_LAB_S/` | Crust + mantle, S-wave incidence (`benchmark/test1_S_out`) |
| `1_Sed_Crust_Mantle_P/` | Sediment + crust + ice cover, P-wave incidence (Antarctic `AT_sed_ice` family) |
| `1_Antarctica_RF/` | Antarctic `AT_crust_ice` profile (ice + crust), full P-wave + RF workflow |
| `2_Scaling_Comparison/` | Python hybrid vs Fortran FDFK2D thread/spatial scaling comparison |

Typical usage (crust–mantle P-wave example):

```bash
conda activate GP245_Final
cd examples/1_Crust_Mantle_P
python run_workflow.py
```

---

## `notebooks/` directory

Top-level Jupyter notebooks (numbered; excludes `old_notebook/`):

| Notebook | Purpose |
|----------|---------|
| `0.0_io_test.ipynb` | I/O performance benchmarks (SU / NPY / NPZ / HDF5) |
| `0.1_rf_compute_test.ipynb` | Receiver-function computation method comparison |
| `1.1_fk_benchmark.ipynb` | FK receiver-function benchmark |
| `1.2_fk_speed.ipynb` | FK propagator speed benchmark |
| `1.3_fk_fft_benchmark.ipynb` | FK FFT backend performance comparison |
| `2.0_fd_validation.ipynb` | FD solver validation (FK reference and analytical Green's function) |
| `2.1_hybrid_vs_fortran_test1_P_incident.ipynb` | Hybrid vs Fortran — test1 P-wave incidence |
| `2.1_hybrid_vs_fortran_test1_S_incident.ipynb` | Hybrid vs Fortran — test1 S-wave incidence |
| `2.2_hybrid_vs_fortran_Altyn.ipynb` | Hybrid vs Fortran — Altyn Tagh profile |

---

## Quick start (Python API)

```python
from fk import FKLayerModel, RickerSource, compute_fk_seismograms

model = FKLayerModel.from_layers(vp=[6000, 8000], vs=[3450, 4450], z_top=[0.0, 35000.0])
source = RickerSource(f0=1.0, incidence="P")
res = compute_fk_seismograms(model, source, rx=[0, 25000, 50000], dt=0.01, nt=4501, p_deg=4.798)
# res.t, res.ux, res.uz
```

For hybrid forward modeling see `hybrid.engine.compute_hybrid_seismograms`; tests: `python -m pytest tests/ -q`.

---

## References

Liu, Y., et al. (2025). FDFK2D: Efficient Two-Dimensional Teleseismic Wavefield Modeling for Receiver Function Analysis Using a Hybrid Method, *Seismol. Res. Lett.* 96, 1163–1180. [doi:10.1785/0220240231](https://doi.org/10.1785/0220240231)

Upstream code: <https://github.com/YoushanLiu/FDFK2D>
