# Benchmark: Altyn Tagh profile

Upstream example: `ref_code/FDFK2D-master_experiment/examples/Altyn`

| Parameter | Value |
|-----------|-------|
| Profile | 0–380 km, depth 0–120 km (real structure) |
| Grid | nx=1901, nz=601, dx=dz=200 m |
| FD order | 6 (half-order 3) |
| Time | tmax=85 s, dt=0.01 s → nt=8501 |
| Source | first event in `Source.dat` (rayp=7.379 s/deg) |

## Submit

```bash
cd benchmark/Altyn
sbatch submit.sbatch
```

Results: `output/benchmark/benchmark_summary.txt`

Expected runtime: several hours (16 CPUs). Request 24 h wall time in `submit.sbatch`.

**Note:** Snapshots are disabled in `input/FD_model.dat` for timing; original is
`input/FD_model.with_snapshots.dat`.
